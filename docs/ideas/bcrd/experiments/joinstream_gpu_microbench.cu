#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Standalone exploratory JoinStream GPU action-space microbenchmark.
//
// Frozen protocol:
//   * exactly one producer and one consumer launch per variant;
//   * K=4 dynamically assigned contributor CTAs;
//   * device-scope release/acquire publication;
//   * 4 tail gaps x 2 residency regimes;
//   * 30 warmups/mode, 30 correctness repeats, 200 utility repeats.
//
// This binary emits raw evidence only.  Final adjudication and the frozen CPU
// Oracle backfeed are intentionally performed by the companion analyzer.

namespace joinstream_gpu {

constexpr const char* kRawSchema = "joinstream-gpu-raw-v1";
constexpr const char* kMetaSchema = "joinstream-gpu-meta-v1";
constexpr int kContributors = 4;
constexpr int kRowElements = 1024;
constexpr int kProjectionOutputs = 64;
constexpr int kProducerThreads = 256;
constexpr int kConsumerThreads = 256;
constexpr int kWarmupsPerMode = 30;
constexpr int kCorrectnessRepeats = 30;
constexpr int kUtilityRepeats = 200;
constexpr std::array<int, 4> kTailGapsUs{{0, 5, 15, 30}};

enum class Variant : int {
  kWholeBarrier = 0,
  kAllDoneSham = 1,
  kJoinStream = 2,
};

enum class WorkMode : int {
  kCorrectness = 0,
  kUtility = 1,
};

const char* VariantName(Variant value) {
  switch (value) {
    case Variant::kWholeBarrier:
      return "A_WholeBarrier";
    case Variant::kAllDoneSham:
      return "B_AllDoneSham";
    case Variant::kJoinStream:
      return "C_JoinStream";
  }
  return "UNKNOWN";
}

const char* ModeName(WorkMode value) {
  return value == WorkMode::kCorrectness ? "correctness" : "utility";
}

struct alignas(128) Control {
  // All atomic_ref targets are naturally aligned POD storage initialized by
  // cudaMemset before each trial.
  alignas(4) uint32_t next_role;
  alignas(4) uint32_t contributors_claimed;
  alignas(4) uint32_t producer_start_gate;
  alignas(4) uint32_t contributions_done;
  alignas(4) uint32_t tail_go;
  alignas(4) uint32_t producer_blocks_done;
  alignas(4) uint32_t consumer_ready;
  alignas(4) uint32_t consumer_entered;

  alignas(8) uint64_t producer_start_ns;
  uint64_t join_close_ns;
  uint64_t row_materialized_ns;
  uint64_t all_blocks_done_ns;
  uint64_t flag_publish_ns;
  uint64_t consumer_entry_ns;
  uint64_t consumer_observe_ns;
  uint64_t consumer_start_ns;
  uint64_t consumer_end_ns;
  uint64_t producer_end_ns;
  uint64_t row_hash;
  uint64_t consumer_hash;
};

static_assert(alignof(Control) >= 8, "Control must preserve atomic alignment");

struct Options {
  int device = 0;
  std::string output_prefix = "joinstream_gpu";
};

struct DeviceBuffers {
  Control* control = nullptr;
  float* contributions = nullptr;
  float* critical_row = nullptr;
  float* projection = nullptr;
  float* tail_sink = nullptr;
  uint64_t* timer_probe = nullptr;
};

struct Cell {
  int index = 0;
  int tail_gap_us = 0;
  std::string residency;
  int producer_blocks = 0;
  uint64_t tail_fma_chunks_per_thread = 0;

  std::string id() const {
    std::ostringstream out;
    out << "tail_us=" << tail_gap_us << "__residency=" << residency;
    return out.str();
  }
};

struct TrialRecord {
  std::string mode;
  std::string cell_id;
  int tail_gap_us = 0;
  std::string residency;
  std::string repeat_kind;
  int repeat_index = 0;
  int permutation_id = 0;
  int permutation_slot = 0;
  std::string sample_order;
  Variant variant = Variant::kWholeBarrier;
  int producer_blocks = 0;
  int producer_block_size = kProducerThreads;
  int consumer_grid_size = 1;
  int consumer_block_size = kConsumerThreads;
  int producer_launches = 1;
  int consumer_launches = 1;
  uint64_t tail_fma_chunks_per_thread = 0;

  uint64_t producer_start_ns = 0;
  uint64_t join_close_ns = 0;
  uint64_t row_materialized_ns = 0;
  uint64_t all_blocks_done_ns = 0;
  uint64_t flag_publish_ns = 0;
  uint64_t consumer_entry_ns = 0;
  uint64_t consumer_observe_ns = 0;
  uint64_t consumer_start_ns = 0;
  uint64_t consumer_end_ns = 0;
  uint64_t producer_end_ns = 0;
  uint64_t total_end_ns = 0;

  uint64_t producer_elapsed_ns = 0;
  uint64_t consumer_end_elapsed_ns = 0;
  uint64_t total_elapsed_ns = 0;
  int64_t visibility_latency_ns = 0;
  int64_t overlap_window_ns = 0;
  int64_t tail_calibration_error_ns = 0;

  uint64_t row_hash = 0;
  uint64_t consumer_hash = 0;
  uint64_t reference_row_hash = 0;
  uint64_t reference_consumer_hash = 0;
  uint64_t input_hash = 0;
  uint64_t work_contract_hash = 0;

  uint32_t contributors_claimed = 0;
  uint32_t expected_contributors = kContributors;
  uint32_t join_counter_final = 0;
  uint32_t blocks_done_final = 0;
  uint32_t expected_blocks = 0;
  uint32_t k_expected = kContributors;

  bool correctness_pass = false;
  bool timestamp_contract_pass = false;
  std::string cuda_error;
};

struct TailCalibration {
  uint64_t rate_probe_chunks = 0;
  uint64_t rate_probe_elapsed_ns = 0;
  double ns_per_chunk = 0.0;
  std::array<uint64_t, kTailGapsUs.size()> chunks_per_thread{};
};

#define CUDA_THROW(call)                                                       \
  do {                                                                         \
    const cudaError_t joinstream_cuda_error = (call);                          \
    if (joinstream_cuda_error != cudaSuccess) {                                \
      std::ostringstream joinstream_cuda_message;                              \
      joinstream_cuda_message << #call << ": "                                \
                              << cudaGetErrorString(joinstream_cuda_error);     \
      throw std::runtime_error(joinstream_cuda_message.str());                 \
    }                                                                          \
  } while (false)

__device__ __forceinline__ uint64_t GlobalTimerNs() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint64_t FnvMix(uint64_t hash, uint32_t value) {
  hash ^= static_cast<uint64_t>(value);
  return hash * 1099511628211ull;
}

__device__ __forceinline__ float ContributionValue(uint32_t contributor,
                                                   uint32_t element) {
  // Exactly representable binary fractions keep the fixed-order materialized
  // row bit-stable across A/B/C.
  const uint32_t integer = (contributor + 1u) * 4096u + (element & 1023u);
  return static_cast<float>(integer) * (1.0f / 1048576.0f);
}

__device__ __forceinline__ float ProjectionWeight(uint32_t output,
                                                  uint32_t element) {
  const int value = static_cast<int>((output * 131u + element * 17u) % 257u) -
                    128;
  return static_cast<float>(value) * (1.0f / 4096.0f);
}

__device__ __forceinline__ float RunFmaChunks(uint64_t chunks,
                                              float accumulator) {
  for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
#pragma unroll
    for (int iteration = 0; iteration < 32; ++iteration) {
      accumulator = fmaf(accumulator, 1.00000011920928955078125f,
                         0.00000011920928955078125f);
    }
  }
  return accumulator;
}

__global__ void TimerProbeKernel(uint64_t* minimum_positive_delta) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint64_t best = ~uint64_t{0};
  uint64_t previous = GlobalTimerNs();
  for (int i = 0; i < 4096; ++i) {
    const uint64_t now = GlobalTimerNs();
    if (now > previous) best = min(best, now - previous);
    previous = now;
  }
  *minimum_positive_delta = best == ~uint64_t{0} ? 0 : best;
}

__global__ void TailCalibrationKernel(uint64_t chunks, uint64_t* elapsed_ns,
                                      float* sink) {
  __shared__ uint64_t start_ns;
  if (threadIdx.x == 0) start_ns = GlobalTimerNs();
  __syncthreads();
  float accumulator =
      1.0f + static_cast<float>(threadIdx.x & 7) * 0.01f;
  accumulator = RunFmaChunks(chunks, accumulator);
  sink[threadIdx.x] = accumulator;
  __syncthreads();
  if (threadIdx.x == 0) *elapsed_ns = GlobalTimerNs() - start_ns;
}

__global__ void ProducerKernel(Control* control, float* contributions,
                               float* critical_row, float* tail_sink,
                               int variant_value,
                               uint64_t tail_fma_chunks_per_thread) {
  __shared__ uint32_t role;
  __shared__ uint32_t is_last_contributor;

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> next_role(
      control->next_role);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> contributors_claimed(
      control->contributors_claimed);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> start_gate(
      control->producer_start_gate);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> contributions_done(
      control->contributions_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_go(control->tail_go);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> blocks_done(
      control->producer_blocks_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumer_ready(
      control->consumer_ready);

  if (threadIdx.x == 0) {
    role = next_role.fetch_add(1u, cuda::memory_order_relaxed);
    if (role < kContributors) {
      contributors_claimed.fetch_add(1u, cuda::memory_order_relaxed);
    }
    if (role == 0) {
      control->producer_start_ns = GlobalTimerNs();
      start_gate.store(1u, cuda::memory_order_release);
    }
  }
  __syncthreads();

  if (role != 0 && threadIdx.x == 0) {
    while (start_gate.load(cuda::memory_order_acquire) == 0u) {
      __nanosleep(64);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) is_last_contributor = 0;
  __syncthreads();
  if (role < kContributors) {
    // The frozen contract deliberately assigns each complete contribution row
    // to contributor thread 0.  Rows do not overlap.
    if (threadIdx.x == 0) {
      float* row = contributions + static_cast<size_t>(role) * kRowElements;
      for (uint32_t element = 0; element < kRowElements; ++element) {
        row[element] = ContributionValue(role, element);
      }

      const uint32_t prior = contributions_done.fetch_add(
          1u, cuda::memory_order_acq_rel);
      is_last_contributor = prior == kContributors - 1 ? 1u : 0u;
      if (is_last_contributor != 0u) {
        // The acquire half of the RMW observes the complete release sequence
        // and therefore all three earlier disjoint contribution rows.
        control->join_close_ns = GlobalTimerNs();
        for (uint32_t element = 0; element < kRowElements; ++element) {
          float value = 0.0f;
#pragma unroll
          for (uint32_t contributor = 0; contributor < kContributors;
               ++contributor) {
            value += contributions[static_cast<size_t>(contributor) *
                                       kRowElements +
                                   element];
          }
          critical_row[element] = value;
        }
        const uint64_t now = GlobalTimerNs();
        control->row_materialized_ns = now;

        if (variant_value == static_cast<int>(Variant::kJoinStream)) {
          control->flag_publish_ns = GlobalTimerNs();
          consumer_ready.store(1u, cuda::memory_order_release);
        }
        // Internal producer-only release.  It is identical in all variants
        // and is not consumed by the consumer kernel.
        tail_go.store(1u, cuda::memory_order_release);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && is_last_contributor == 0u) {
    while (tail_go.load(cuda::memory_order_acquire) == 0u) {
      __nanosleep(64);
    }
  }
  __syncthreads();

  // Frozen finite residual work.  Startup calibration maps each target gap to
  // one fixed chunk count.  Every A/B/C producer thread executes exactly that
  // many chunks (32 FMAs/chunk), independent of runtime scheduling or timers.
  float accumulator = 1.0f + static_cast<float>(threadIdx.x & 7) * 0.01f;
  accumulator = RunFmaChunks(tail_fma_chunks_per_thread, accumulator);
  tail_sink[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
      accumulator;
  __syncthreads();

  if (threadIdx.x == 0) {
    const uint32_t prior =
        blocks_done.fetch_add(1u, cuda::memory_order_acq_rel);
    if (prior == gridDim.x - 1) {
      control->all_blocks_done_ns = GlobalTimerNs();
      if (variant_value == static_cast<int>(Variant::kAllDoneSham)) {
        control->flag_publish_ns = GlobalTimerNs();
        consumer_ready.store(1u, cuda::memory_order_release);
      }
      control->producer_end_ns = GlobalTimerNs();
    }
  }
}

__device__ uint64_t HashCriticalRow(const float* critical_row) {
  uint64_t hash = 1469598103934665603ull;
  float reduction = 0.0f;
  for (int element = 0; element < kRowElements; ++element) {
    const float value = critical_row[element];
    hash = FnvMix(hash, __float_as_uint(value));
    reduction += value;
  }
  return FnvMix(hash, __float_as_uint(reduction));
}

__global__ void ConsumerKernel(Control* control, const float* critical_row,
                               float* projection, int variant_value,
                               int mode_value) {
  __shared__ float reduction[kConsumerThreads];
  __shared__ float outputs[kProjectionOutputs];
  __shared__ float inverse_rms;
  __shared__ int top_one;
  __shared__ int top_two;

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumer_ready(
      control->consumer_ready);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> consumer_entered(
      control->consumer_entered);

  if (threadIdx.x == 0) {
    control->consumer_entry_ns = GlobalTimerNs();
    if (variant_value != static_cast<int>(Variant::kWholeBarrier)) {
      // Host observes this with pinned D2H copies on an independent nonblocking
      // copy stream before it is allowed to enqueue the B/C producer.
      __threadfence_system();
      consumer_entered.store(1u, cuda::memory_order_release);
      __threadfence_system();
      while (consumer_ready.load(cuda::memory_order_acquire) == 0u) {
        __nanosleep(64);
      }
      control->consumer_observe_ns = GlobalTimerNs();
    }
  }
  __syncthreads();

  if (variant_value != static_cast<int>(Variant::kWholeBarrier)) {
    // Every thread that may read critical_row performs an acquire.  This is
    // intentionally stronger than relying on lane 0 plus a CTA barrier.
    (void)consumer_ready.load(cuda::memory_order_acquire);
  }
  __syncthreads();

  // This is the final CTA barrier before useful work.  Other lanes may begin a
  // few cycles before lane 0's timestamp store, making the recorded start
  // conservatively late; no subsequent barrier can make it artificially early.
  if (threadIdx.x == 0) control->consumer_start_ns = GlobalTimerNs();

  uint64_t row_hash = 0;
  if (threadIdx.x == 0) row_hash = HashCriticalRow(critical_row);

  uint64_t work_hash = 0;
  if (mode_value == static_cast<int>(WorkMode::kCorrectness)) {
    if (threadIdx.x == 0) work_hash = row_hash;
  } else {
    float local_sum = 0.0f;
    for (int element = threadIdx.x; element < kRowElements;
         element += blockDim.x) {
      const float value = critical_row[element];
      local_sum = fmaf(value, value, local_sum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      inverse_rms = rsqrtf(reduction[0] / static_cast<float>(kRowElements) +
                           1.0e-5f);
    }
    __syncthreads();

    if (threadIdx.x < kProjectionOutputs) {
      float projected = 0.0f;
      for (int element = 0; element < kRowElements; ++element) {
        projected = fmaf(critical_row[element] * inverse_rms,
                         ProjectionWeight(threadIdx.x, element), projected);
      }
      outputs[threadIdx.x] = projected;
      projection[threadIdx.x] = projected;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      top_one = 0;
      top_two = 1;
      if (outputs[top_two] > outputs[top_one]) {
        const int swap = top_one;
        top_one = top_two;
        top_two = swap;
      }
      for (int output = 2; output < kProjectionOutputs; ++output) {
        if (outputs[output] > outputs[top_one]) {
          top_two = top_one;
          top_one = output;
        } else if (outputs[output] > outputs[top_two]) {
          top_two = output;
        }
      }
      work_hash = row_hash;
      for (int output = 0; output < kProjectionOutputs; ++output) {
        work_hash = FnvMix(work_hash, __float_as_uint(outputs[output]));
      }
      work_hash = FnvMix(work_hash, static_cast<uint32_t>(top_one));
      work_hash = FnvMix(work_hash, static_cast<uint32_t>(top_two));
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    control->row_hash = row_hash;
    control->consumer_hash = work_hash;
    control->consumer_end_ns = GlobalTimerNs();
  }
}

uint64_t FnvString(const std::string& value) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char byte : value) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char byte : value) {
    switch (byte) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (byte < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(byte) << std::dec;
        } else {
          out << static_cast<char>(byte);
        }
    }
  }
  return out.str();
}

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char byte : value) {
    if (byte == '"') escaped += '"';
    escaped += byte;
  }
  escaped += '"';
  return escaped;
}

uint64_t SafeElapsed(uint64_t end, uint64_t start) {
  return end >= start ? end - start : 0;
}

int64_t SignedDifference(uint64_t lhs, uint64_t rhs) {
  if (lhs >= rhs) {
    const uint64_t delta = lhs - rhs;
    return delta > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
               ? std::numeric_limits<int64_t>::max()
               : static_cast<int64_t>(delta);
  }
  const uint64_t delta = rhs - lhs;
  return delta > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
             ? std::numeric_limits<int64_t>::min()
             : -static_cast<int64_t>(delta);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--device N] [--output-prefix PATH]\n";
      std::exit(0);
    }
    if (argument == "--device" && index + 1 < argc) {
      options.device = std::stoi(argv[++index]);
    } else if (argument == "--output-prefix" && index + 1 < argc) {
      options.output_prefix = argv[++index];
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + argument);
    }
  }
  if (options.output_prefix.empty()) {
    throw std::invalid_argument("--output-prefix must not be empty");
  }
  return options;
}

void AllocateBuffers(DeviceBuffers* buffers, int maximum_producer_blocks) {
  CUDA_THROW(cudaMalloc(&buffers->control, sizeof(Control)));
  CUDA_THROW(cudaMalloc(&buffers->contributions,
                        sizeof(float) * kContributors * kRowElements));
  CUDA_THROW(cudaMalloc(&buffers->critical_row,
                        sizeof(float) * kRowElements));
  CUDA_THROW(cudaMalloc(&buffers->projection,
                        sizeof(float) * kProjectionOutputs));
  CUDA_THROW(cudaMalloc(&buffers->tail_sink,
                        sizeof(float) * maximum_producer_blocks *
                            kProducerThreads));
  CUDA_THROW(cudaMalloc(&buffers->timer_probe, sizeof(uint64_t)));
}

void FreeBuffers(DeviceBuffers* buffers) {
  cudaFree(buffers->timer_probe);
  cudaFree(buffers->tail_sink);
  cudaFree(buffers->projection);
  cudaFree(buffers->critical_row);
  cudaFree(buffers->contributions);
  cudaFree(buffers->control);
  *buffers = DeviceBuffers{};
}

void ResetBuffers(const DeviceBuffers& buffers, int maximum_producer_blocks) {
  CUDA_THROW(cudaMemset(buffers.control, 0, sizeof(Control)));
  CUDA_THROW(cudaMemset(buffers.contributions, 0,
                        sizeof(float) * kContributors * kRowElements));
  CUDA_THROW(cudaMemset(buffers.critical_row, 0,
                        sizeof(float) * kRowElements));
  CUDA_THROW(cudaMemset(buffers.projection, 0,
                        sizeof(float) * kProjectionOutputs));
  CUDA_THROW(cudaMemset(buffers.tail_sink, 0,
                        sizeof(float) * maximum_producer_blocks *
                            kProducerThreads));
  CUDA_THROW(cudaDeviceSynchronize());
}

bool TimestampContract(const TrialRecord& row) {
  if (row.producer_start_ns == 0 || row.join_close_ns < row.producer_start_ns ||
      row.row_materialized_ns < row.join_close_ns ||
      row.all_blocks_done_ns < row.row_materialized_ns ||
      row.producer_end_ns < row.all_blocks_done_ns ||
      row.consumer_entry_ns == 0 ||
      row.consumer_start_ns < row.consumer_entry_ns ||
      row.consumer_end_ns < row.consumer_start_ns) {
    return false;
  }
  if (row.contributors_claimed != row.expected_contributors ||
      row.join_counter_final != row.expected_contributors ||
      row.blocks_done_final != row.expected_blocks) {
    return false;
  }
  if (row.variant == Variant::kWholeBarrier) {
    return row.flag_publish_ns == 0 && row.consumer_observe_ns == 0 &&
           row.consumer_entry_ns >= row.producer_end_ns;
  }
  if (row.flag_publish_ns == 0 ||
      row.consumer_observe_ns < row.flag_publish_ns ||
      row.consumer_start_ns < row.consumer_observe_ns ||
      row.consumer_entry_ns > row.producer_start_ns) {
    return false;
  }
  if (row.variant == Variant::kAllDoneSham) {
    return row.flag_publish_ns >= row.all_blocks_done_ns;
  }
  return row.flag_publish_ns >= row.row_materialized_ns;
}

void WaitForConsumerEntry(const DeviceBuffers& buffers,
                          cudaStream_t copy_stream,
                          uint32_t* pinned_consumer_entered) {
  *pinned_consumer_entered = 0u;
  const auto* device_base =
      reinterpret_cast<const unsigned char*>(buffers.control);
  const void* device_entered =
      device_base + offsetof(Control, consumer_entered);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  while (*pinned_consumer_entered == 0u) {
    CUDA_THROW(cudaMemcpyAsync(pinned_consumer_entered, device_entered,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               copy_stream));
    CUDA_THROW(cudaStreamSynchronize(copy_stream));
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
          "consumer entry handshake timed out before producer launch");
    }
  }
}

TrialRecord RunTrial(const DeviceBuffers& buffers, int maximum_producer_blocks,
                     const Cell& cell, WorkMode mode,
                     const std::string& repeat_kind, int repeat_index,
                     int permutation_id, int permutation_slot,
                     const std::string& sample_order, Variant variant,
                     cudaStream_t producer_stream, cudaStream_t consumer_stream,
                     cudaStream_t copy_stream,
                     cudaEvent_t producer_done_event,
                     uint32_t* pinned_consumer_entered,
                     uint64_t input_hash,
                     uint64_t work_contract_hash) {
  ResetBuffers(buffers, maximum_producer_blocks);

  const uint64_t tail_gap_ns = static_cast<uint64_t>(cell.tail_gap_us) * 1000ull;
  if (variant == Variant::kWholeBarrier) {
    ProducerKernel<<<cell.producer_blocks, kProducerThreads, 0, producer_stream>>>(
        buffers.control, buffers.contributions, buffers.critical_row,
        buffers.tail_sink, static_cast<int>(variant),
        cell.tail_fma_chunks_per_thread);
    CUDA_THROW(cudaGetLastError());
    CUDA_THROW(cudaEventRecord(producer_done_event, producer_stream));
    CUDA_THROW(cudaStreamWaitEvent(consumer_stream, producer_done_event, 0));
    ConsumerKernel<<<1, kConsumerThreads, 0, consumer_stream>>>(
        buffers.control, buffers.critical_row, buffers.projection,
        static_cast<int>(variant), static_cast<int>(mode));
    CUDA_THROW(cudaGetLastError());
  } else {
    // B and C have exactly the same pre-launched persistent consumer structure.
    ConsumerKernel<<<1, kConsumerThreads, 0, consumer_stream>>>(
        buffers.control, buffers.critical_row, buffers.projection,
        static_cast<int>(variant), static_cast<int>(mode));
    CUDA_THROW(cudaGetLastError());
    WaitForConsumerEntry(buffers, copy_stream, pinned_consumer_entered);
    ProducerKernel<<<cell.producer_blocks, kProducerThreads, 0, producer_stream>>>(
        buffers.control, buffers.contributions, buffers.critical_row,
        buffers.tail_sink, static_cast<int>(variant),
        cell.tail_fma_chunks_per_thread);
    CUDA_THROW(cudaGetLastError());
  }
  CUDA_THROW(cudaDeviceSynchronize());

  Control control{};
  CUDA_THROW(cudaMemcpy(&control, buffers.control, sizeof(Control),
                        cudaMemcpyDeviceToHost));

  TrialRecord row;
  row.mode = ModeName(mode);
  row.cell_id = cell.id();
  row.tail_gap_us = cell.tail_gap_us;
  row.residency = cell.residency;
  row.repeat_kind = repeat_kind;
  row.repeat_index = repeat_index;
  row.permutation_id = permutation_id;
  row.permutation_slot = permutation_slot;
  row.sample_order = sample_order;
  row.variant = variant;
  row.producer_blocks = cell.producer_blocks;
  row.expected_blocks = static_cast<uint32_t>(cell.producer_blocks);
  row.tail_fma_chunks_per_thread = cell.tail_fma_chunks_per_thread;

  row.producer_start_ns = control.producer_start_ns;
  row.join_close_ns = control.join_close_ns;
  row.row_materialized_ns = control.row_materialized_ns;
  row.all_blocks_done_ns = control.all_blocks_done_ns;
  row.flag_publish_ns = control.flag_publish_ns;
  row.consumer_entry_ns = control.consumer_entry_ns;
  row.consumer_observe_ns = control.consumer_observe_ns;
  row.consumer_start_ns = control.consumer_start_ns;
  row.consumer_end_ns = control.consumer_end_ns;
  row.producer_end_ns = control.producer_end_ns;
  row.total_end_ns = std::max(row.producer_end_ns, row.consumer_end_ns);

  row.producer_elapsed_ns =
      SafeElapsed(row.producer_end_ns, row.producer_start_ns);
  row.consumer_end_elapsed_ns =
      SafeElapsed(row.consumer_end_ns, row.producer_start_ns);
  row.total_elapsed_ns = SafeElapsed(row.total_end_ns, row.producer_start_ns);
  row.visibility_latency_ns =
      variant == Variant::kWholeBarrier
          ? 0
          : SignedDifference(row.consumer_observe_ns, row.flag_publish_ns);
  row.overlap_window_ns =
      SignedDifference(row.producer_end_ns, row.consumer_start_ns);
  row.tail_calibration_error_ns =
      SignedDifference(row.producer_end_ns, row.row_materialized_ns) -
      static_cast<int64_t>(tail_gap_ns);

  row.row_hash = control.row_hash;
  row.consumer_hash = control.consumer_hash;
  row.input_hash = input_hash;
  row.work_contract_hash = work_contract_hash;
  row.contributors_claimed = control.contributors_claimed;
  row.join_counter_final = control.contributions_done;
  row.blocks_done_final = control.producer_blocks_done;
  row.timestamp_contract_pass = TimestampContract(row);
  return row;
}

void WriteCsvHeader(std::ostream& out) {
  out << "schema_version,mode,cell_id,tail_gap_us,tail_fma_chunks_per_thread,"
         "residency,repeat_kind,"
         "repeat_index,permutation_id,permutation_slot,sample_order,variant,"
         "producer_blocks,producer_block_size,consumer_block_size,"
         "consumer_grid_size,"
         "producer_launches,consumer_launches,producer_start_ns,join_close_ns,"
         "row_materialized_ns,all_blocks_done_ns,flag_publish_ns,"
         "consumer_entry_ns,consumer_observe_ns,consumer_start_ns,"
         "consumer_end_ns,producer_end_ns,total_end_ns,producer_elapsed_ns,"
         "consumer_end_elapsed_ns,total_elapsed_ns,visibility_latency_ns,"
         "overlap_window_ns,tail_calibration_error_ns,row_hash,consumer_hash,"
         "reference_row_hash,reference_consumer_hash,input_hash,"
         "work_contract_hash,contributors_claimed,expected_contributors,"
         "join_counter_final,blocks_done_final,expected_blocks,k_expected,"
         "correctness_pass,timestamp_contract_pass,cuda_error\n";
}

void WriteCsvRow(std::ostream& out, const TrialRecord& row) {
  out << kRawSchema << ',' << row.mode << ',' << CsvEscape(row.cell_id) << ','
      << row.tail_gap_us << ',' << row.tail_fma_chunks_per_thread << ','
      << row.residency << ',' << row.repeat_kind << ','
      << row.repeat_index << ',' << row.permutation_id << ','
      << row.permutation_slot << ',' << row.sample_order << ','
      << VariantName(row.variant) << ',' << row.producer_blocks << ','
      << row.producer_block_size << ',' << row.consumer_block_size << ','
      << row.consumer_grid_size << ','
      << row.producer_launches << ',' << row.consumer_launches << ','
      << row.producer_start_ns << ',' << row.join_close_ns << ','
      << row.row_materialized_ns << ',' << row.all_blocks_done_ns << ','
      << row.flag_publish_ns << ',' << row.consumer_entry_ns << ','
      << row.consumer_observe_ns << ',' << row.consumer_start_ns << ','
      << row.consumer_end_ns << ',' << row.producer_end_ns << ','
      << row.total_end_ns << ',' << row.producer_elapsed_ns << ','
      << row.consumer_end_elapsed_ns << ',' << row.total_elapsed_ns << ','
      << row.visibility_latency_ns << ',' << row.overlap_window_ns << ','
      << row.tail_calibration_error_ns << ',' << row.row_hash << ','
      << row.consumer_hash << ',' << row.reference_row_hash << ','
      << row.reference_consumer_hash << ',' << row.input_hash << ','
      << row.work_contract_hash << ',' << row.contributors_claimed << ','
      << row.expected_contributors << ',' << row.join_counter_final << ','
      << row.blocks_done_final << ',' << row.expected_blocks << ','
      << row.k_expected << ',' << (row.correctness_pass ? 1 : 0) << ','
      << (row.timestamp_contract_pass ? 1 : 0) << ','
      << CsvEscape(row.cuda_error) << '\n';
}

std::string PermutationName(const std::array<Variant, 3>& permutation) {
  std::string value;
  for (Variant variant : permutation) {
    value.push_back(variant == Variant::kWholeBarrier
                        ? 'A'
                        : (variant == Variant::kAllDoneSham ? 'B' : 'C'));
  }
  return value;
}

const std::array<std::array<Variant, 3>, 6>& Permutations() {
  static const std::array<std::array<Variant, 3>, 6> permutations{{
      {{Variant::kWholeBarrier, Variant::kAllDoneSham,
        Variant::kJoinStream}},
      {{Variant::kWholeBarrier, Variant::kJoinStream,
        Variant::kAllDoneSham}},
      {{Variant::kAllDoneSham, Variant::kWholeBarrier,
        Variant::kJoinStream}},
      {{Variant::kAllDoneSham, Variant::kJoinStream,
        Variant::kWholeBarrier}},
      {{Variant::kJoinStream, Variant::kWholeBarrier,
        Variant::kAllDoneSham}},
      {{Variant::kJoinStream, Variant::kAllDoneSham,
        Variant::kWholeBarrier}},
  }};
  return permutations;
}

int VariantIndex(Variant variant) { return static_cast<int>(variant); }

uint64_t MeasureTimerResolution(const DeviceBuffers& buffers) {
  CUDA_THROW(cudaMemset(buffers.timer_probe, 0, sizeof(uint64_t)));
  TimerProbeKernel<<<1, 1>>>(buffers.timer_probe);
  CUDA_THROW(cudaGetLastError());
  CUDA_THROW(cudaDeviceSynchronize());
  uint64_t resolution = 0;
  CUDA_THROW(cudaMemcpy(&resolution, buffers.timer_probe, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  return resolution;
}

TailCalibration CalibrateTailChunks(const DeviceBuffers& buffers) {
  constexpr uint64_t kRateProbeChunks = 16384;
  // Two untimed prewarm launches ensure the rate probe itself is not measuring
  // first-launch or cold instruction-cache effects.
  for (int warmup = 0; warmup < 2; ++warmup) {
    TailCalibrationKernel<<<1, kProducerThreads>>>(
        kRateProbeChunks, buffers.timer_probe, buffers.tail_sink);
    CUDA_THROW(cudaGetLastError());
    CUDA_THROW(cudaDeviceSynchronize());
  }

  std::array<uint64_t, 5> samples{};
  for (uint64_t& sample : samples) {
    CUDA_THROW(cudaMemset(buffers.timer_probe, 0, sizeof(uint64_t)));
    TailCalibrationKernel<<<1, kProducerThreads>>>(
        kRateProbeChunks, buffers.timer_probe, buffers.tail_sink);
    CUDA_THROW(cudaGetLastError());
    CUDA_THROW(cudaDeviceSynchronize());
    CUDA_THROW(cudaMemcpy(&sample, buffers.timer_probe, sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    if (sample == 0) {
      throw std::runtime_error("tail FMA calibration returned zero elapsed time");
    }
  }
  std::sort(samples.begin(), samples.end());

  TailCalibration calibration;
  calibration.rate_probe_chunks = kRateProbeChunks;
  calibration.rate_probe_elapsed_ns = samples[samples.size() / 2];
  calibration.ns_per_chunk =
      static_cast<double>(calibration.rate_probe_elapsed_ns) /
      static_cast<double>(kRateProbeChunks);
  if (!(calibration.ns_per_chunk > 0.0) ||
      !std::isfinite(calibration.ns_per_chunk)) {
    throw std::runtime_error("invalid tail FMA calibration rate");
  }
  for (size_t index = 0; index < kTailGapsUs.size(); ++index) {
    if (kTailGapsUs[index] == 0) {
      calibration.chunks_per_thread[index] = 0;
      continue;
    }
    const double target_ns = static_cast<double>(kTailGapsUs[index]) * 1000.0;
    calibration.chunks_per_thread[index] = std::max<uint64_t>(
        1u, static_cast<uint64_t>(std::llround(target_ns /
                                              calibration.ns_per_chunk)));
    // Prewarm the exact frozen chunk count once.  It is not adjusted from the
    // observed result, so A/B/C later receive one immutable work amount.
    TailCalibrationKernel<<<1, kProducerThreads>>>(
        calibration.chunks_per_thread[index], buffers.timer_probe,
        buffers.tail_sink);
    CUDA_THROW(cudaGetLastError());
    CUDA_THROW(cudaDeviceSynchronize());
  }
  return calibration;
}

void WriteBlockedMeta(const std::string& path, const std::string& raw_path,
                      int requested_device, const std::string& reason) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open meta output: " + path);
  out << "{\n"
      << "  \"schema\": \"" << kMetaSchema << "\",\n"
      << "  \"status\": \"BLOCKED_NO_GPU\",\n"
      << "  \"verdict_pending\": false,\n"
      << "  \"files\": {\"raw_csv\": \"" << JsonEscape(raw_path)
      << "\"},\n"
      << "  \"requested_device\": " << requested_device << ",\n"
      << "  \"reason\": \"" << JsonEscape(reason) << "\",\n"
      << "  \"timer\": {\"source\": \"ptx_%globaltimer\", "
         "\"resolution_ns\": 0, "
         "\"cross_variant_comparison\": "
         "\"per_trial_elapsed_from_producer_start\"}\n"
      << "}\n";
}

void WriteCompletedMeta(const std::string& path, const std::string& raw_path,
                        const cudaDeviceProp& property, int device,
                        int driver_version, int runtime_version,
                        int least_priority, int greatest_priority,
                        int active_blocks_per_sm, int friendly_blocks,
                        int saturating_blocks,
                        const TailCalibration& tail_calibration,
                        uint64_t timer_resolution_ns,
                        uint64_t input_contract_hash,
                        uint64_t work_contract_hash, uint64_t emitted_rows,
                        uint64_t contract_failures) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open meta output: " + path);
  out << "{\n"
      << "  \"schema\": \"" << kMetaSchema << "\",\n"
      << "  \"status\": \"COMPLETED_RAW_NOT_ADJUDICATED\",\n"
      << "  \"verdict_pending\": true,\n"
      << "  \"files\": {\"raw_csv\": \"" << JsonEscape(raw_path)
      << "\"},\n"
      << "  \"protocol\": {\n"
      << "    \"variants\": [\"A_WholeBarrier\", \"B_AllDoneSham\", "
         "\"C_JoinStream\"],\n"
      << "    \"k_contributors\": " << kContributors << ",\n"
      << "    \"row_elements\": " << kRowElements << ",\n"
      << "    \"projection_outputs\": " << kProjectionOutputs << ",\n"
      << "    \"tail_gap_us\": [0, 5, 15, 30],\n"
      << "    \"residency\": [\"tail-friendly\", "
         "\"near-saturating\"],\n"
      << "    \"warmups_per_cell_mode\": " << kWarmupsPerMode << ",\n"
      << "    \"correctness_repeats\": " << kCorrectnessRepeats << ",\n"
      << "    \"utility_repeats\": " << kUtilityRepeats << ",\n"
      << "    \"producer_launches_per_trial\": 1,\n"
      << "    \"consumer_launches_per_trial\": 1,\n"
      << "    \"bc_consumer_entry_handshake\": "
         "\"pinned_d2h_copy_stream_before_producer_enqueue\"\n"
      << "  },\n"
      << "  \"tail_fma_calibration\": {\n"
      << "    \"fmas_per_chunk_per_thread\": 32,\n"
      << "    \"rate_probe_chunks\": " << tail_calibration.rate_probe_chunks
      << ",\n"
      << "    \"rate_probe_elapsed_ns\": "
      << tail_calibration.rate_probe_elapsed_ns << ",\n"
      << "    \"ns_per_chunk\": " << std::setprecision(12)
      << tail_calibration.ns_per_chunk << ",\n"
      << "    \"targets\": [\n"
      << "      {\"tail_gap_us\": 0, \"chunks_per_thread\": "
      << tail_calibration.chunks_per_thread[0] << "},\n"
      << "      {\"tail_gap_us\": 5, \"chunks_per_thread\": "
      << tail_calibration.chunks_per_thread[1] << "},\n"
      << "      {\"tail_gap_us\": 15, \"chunks_per_thread\": "
      << tail_calibration.chunks_per_thread[2] << "},\n"
      << "      {\"tail_gap_us\": 30, \"chunks_per_thread\": "
      << tail_calibration.chunks_per_thread[3] << "}\n"
      << "    ]\n"
      << "  },\n"
      << "  \"hardware\": {\n"
      << "    \"device_index\": " << device << ",\n"
      << "    \"name\": \"" << JsonEscape(property.name) << "\",\n"
      << "    \"compute_capability\": \"" << property.major << '.'
      << property.minor << "\",\n"
      << "    \"sm_count\": " << property.multiProcessorCount << ",\n"
      << "    \"total_global_memory_bytes\": " << property.totalGlobalMem
      << "\n  },\n"
      << "  \"software\": {\"cuda_driver_version\": " << driver_version
      << ", \"cuda_runtime_version\": " << runtime_version << "},\n"
      << "  \"clock_config\": {\"reported_core_clock_khz\": "
      << property.clockRate << ", \"reported_memory_clock_khz\": "
      << property.memoryClockRate
      << ", \"application_clocks_locked_by_benchmark\": false},\n"
      << "  \"stream_priorities\": {\"producer\": " << least_priority
      << ", \"consumer\": " << greatest_priority << "},\n"
      << "  \"launch_geometry\": {\n"
      << "    \"producer_block_size\": " << kProducerThreads << ",\n"
      << "    \"consumer_block_size\": " << kConsumerThreads << ",\n"
      << "    \"producer_active_blocks_per_sm\": " << active_blocks_per_sm
      << ",\n"
      << "    \"tail_friendly_blocks\": " << friendly_blocks << ",\n"
      << "    \"near_saturating_blocks\": " << saturating_blocks << "\n"
      << "  },\n"
      << "  \"timer\": {\"source\": \"ptx_%globaltimer\", "
         "\"resolution_ns\": "
      << timer_resolution_ns
      << ", \"cross_variant_comparison\": "
         "\"per_trial_elapsed_from_producer_start\"},\n"
      << "  \"contracts\": {\"input_contract_hash\": "
      << input_contract_hash << ", \"work_contract_hash\": "
      << work_contract_hash << "},\n"
      << "  \"counts\": {\"cells\": 8, \"csv_rows\": " << emitted_rows
      << ", \"contract_failures\": " << contract_failures << "}\n"
      << "}\n";
}

}  // namespace joinstream_gpu

int main(int argc, char** argv) {
  using namespace joinstream_gpu;
  std::string raw_path;
  std::string meta_path;
  try {
    const Options options = ParseOptions(argc, argv);
    raw_path = options.output_prefix + ".csv";
    meta_path = options.output_prefix + ".meta.json";

    // Always create a schema-bearing CSV, including BLOCKED_NO_GPU runs.
    std::ofstream csv(raw_path);
    if (!csv) throw std::runtime_error("cannot open raw output: " + raw_path);
    WriteCsvHeader(csv);

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
      const std::string reason =
          count_status == cudaSuccess ? "cudaGetDeviceCount returned zero"
                                      : cudaGetErrorString(count_status);
      WriteBlockedMeta(meta_path, raw_path, options.device, reason);
      std::cerr << "BLOCKED_NO_GPU: " << reason << '\n';
      return 2;
    }
    if (options.device < 0 || options.device >= device_count) {
      WriteBlockedMeta(meta_path, raw_path, options.device,
                       "requested device index is unavailable");
      std::cerr << "BLOCKED_NO_GPU: requested device index is unavailable\n";
      return 2;
    }

    CUDA_THROW(cudaSetDevice(options.device));
    cudaDeviceProp property{};
    CUDA_THROW(cudaGetDeviceProperties(&property, options.device));

    int active_blocks_per_sm = 0;
    CUDA_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, ProducerKernel, kProducerThreads, 0));
    if (active_blocks_per_sm <= 0) {
      throw std::runtime_error("producer occupancy is zero");
    }
    const int saturating_blocks =
        property.multiProcessorCount * active_blocks_per_sm;
    const int friendly_blocks =
        std::max(kContributors, property.multiProcessorCount / 2);
    if (saturating_blocks < kContributors ||
        friendly_blocks >= saturating_blocks) {
      throw std::runtime_error(
          "GPU cannot express both frozen residency regimes with K=4");
    }

    int least_priority = 0;
    int greatest_priority = 0;
    CUDA_THROW(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    cudaStream_t producer_stream = nullptr;
    cudaStream_t consumer_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t producer_done_event = nullptr;
    uint32_t* pinned_consumer_entered = nullptr;
    CUDA_THROW(cudaStreamCreateWithPriority(
        &producer_stream, cudaStreamNonBlocking, least_priority));
    CUDA_THROW(cudaStreamCreateWithPriority(
        &consumer_stream, cudaStreamNonBlocking, greatest_priority));
    CUDA_THROW(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_THROW(cudaHostAlloc(&pinned_consumer_entered, sizeof(uint32_t),
                             cudaHostAllocDefault));
    CUDA_THROW(
        cudaEventCreateWithFlags(&producer_done_event, cudaEventDisableTiming));

    DeviceBuffers buffers;
    AllocateBuffers(&buffers, saturating_blocks);
    const uint64_t timer_resolution_ns = MeasureTimerResolution(buffers);
    const TailCalibration tail_calibration = CalibrateTailChunks(buffers);

    const uint64_t input_contract_hash = FnvString(
        "contribution_value:v1;K=4;row_elements=1024;disjoint_rows");
    const uint64_t work_contract_hash = FnvString(
        "consumer:v1;row_hash;RMSNorm;projection=64;topk=2;block=256");

    std::vector<Cell> cells;
    int cell_index = 0;
    for (size_t tail_index = 0; tail_index < kTailGapsUs.size(); ++tail_index) {
      const int tail_gap_us = kTailGapsUs[tail_index];
      const uint64_t tail_chunks =
          tail_calibration.chunks_per_thread[tail_index];
      cells.push_back(Cell{cell_index++, tail_gap_us, "tail-friendly",
                           friendly_blocks, tail_chunks});
      cells.push_back(Cell{cell_index++, tail_gap_us, "near-saturating",
                           saturating_blocks, tail_chunks});
    }

    uint64_t emitted_rows = 0;
    uint64_t contract_failures = 0;
    const auto& permutations = Permutations();
    for (const Cell& cell : cells) {
      for (WorkMode mode : {WorkMode::kCorrectness, WorkMode::kUtility}) {
        for (int phase = 0; phase < 2; ++phase) {
          const bool warmup = phase == 0;
          const int repeats =
              warmup ? kWarmupsPerMode
                     : (mode == WorkMode::kCorrectness ? kCorrectnessRepeats
                                                        : kUtilityRepeats);
          const std::string repeat_kind = warmup ? "warmup" : "measured";
          for (int repeat = 0; repeat < repeats; ++repeat) {
            const int permutation_id =
                repeat % static_cast<int>(permutations.size());
            const auto& permutation = permutations[permutation_id];
            const std::string sample_order = PermutationName(permutation);
            std::array<TrialRecord, 3> rows;
            for (int slot = 0; slot < 3; ++slot) {
              const Variant variant = permutation[slot];
              TrialRecord row = RunTrial(
                  buffers, saturating_blocks, cell, mode, repeat_kind, repeat,
                  permutation_id, slot, sample_order, variant, producer_stream,
                  consumer_stream, copy_stream, producer_done_event,
                  pinned_consumer_entered, input_contract_hash,
                  work_contract_hash);
              rows[VariantIndex(variant)] = std::move(row);
            }

            const uint64_t reference_row_hash =
                rows[VariantIndex(Variant::kWholeBarrier)].row_hash;
            const uint64_t reference_consumer_hash =
                rows[VariantIndex(Variant::kWholeBarrier)].consumer_hash;
            for (TrialRecord& row : rows) {
              row.reference_row_hash = reference_row_hash;
              row.reference_consumer_hash = reference_consumer_hash;
              row.correctness_pass =
                  row.timestamp_contract_pass && row.cuda_error.empty() &&
                  row.row_hash == reference_row_hash &&
                  row.consumer_hash == reference_consumer_hash &&
                  reference_row_hash != 0 && reference_consumer_hash != 0;
              if (!row.correctness_pass) ++contract_failures;
              WriteCsvRow(csv, row);
              ++emitted_rows;
            }
            csv.flush();
          }
        }
      }
      std::cout << "completed " << cell.id() << '\n';
    }

    int driver_version = 0;
    int runtime_version = 0;
    CUDA_THROW(cudaDriverGetVersion(&driver_version));
    CUDA_THROW(cudaRuntimeGetVersion(&runtime_version));
    WriteCompletedMeta(meta_path, raw_path, property, options.device,
                       driver_version, runtime_version, least_priority,
                       greatest_priority, active_blocks_per_sm, friendly_blocks,
                       saturating_blocks, tail_calibration,
                       timer_resolution_ns,
                       input_contract_hash, work_contract_hash, emitted_rows,
                       contract_failures);

    FreeBuffers(&buffers);
    cudaFreeHost(pinned_consumer_entered);
    cudaEventDestroy(producer_done_event);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(consumer_stream);
    cudaStreamDestroy(producer_stream);

    std::cout << "raw_csv=" << raw_path << '\n'
              << "meta_json=" << meta_path << '\n'
              << "status=COMPLETED_RAW_NOT_ADJUDICATED\n";
    return contract_failures == 0 ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "INVALID_MEMORY_CONTRACT: " << error.what() << '\n';
    if (!meta_path.empty()) {
      try {
        std::ofstream out(meta_path);
        if (out) {
          out << "{\n  \"schema\": \"joinstream-gpu-meta-v1\",\n"
              << "  \"status\": \"INVALID_MEMORY_CONTRACT\",\n"
              << "  \"verdict_pending\": false,\n"
              << "  \"reason\": \"" << JsonEscape(error.what()) << "\"\n}\n";
        }
      } catch (...) {
      }
    }
    return 3;
  }
}
