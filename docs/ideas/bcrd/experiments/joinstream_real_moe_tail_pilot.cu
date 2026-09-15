#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// REAL_MOE_TAIL_APPLICABILITY_GATE
//
// A standalone, dependency-free grouped-expert CUDA pilot.  The formal
// producer contains only route-indexed FP16 matrix multiplication and progress
// instrumentation: no sleep, clock-spin, empty delay, SM reservation, or
// producer split.  Calibration and formal route seeds are mechanically
// separated by a run lock.

namespace real_moe_tail {

constexpr const char* kRawSchema = "joinstream-real-moe-tail-raw-v1";
constexpr const char* kMetaSchema = "joinstream-real-moe-tail-meta-v1";
constexpr const char* kCalibrationSchema =
    "joinstream-real-moe-tail-calibration-v1";
constexpr const char* kRunLockSchema = "joinstream-real-moe-tail-run-lock-v1";
constexpr const char* kEnvironmentSchema =
    "joinstream-real-moe-tail-environment-v1";
constexpr const char* kPreflightSchema =
    "joinstream-real-moe-tail-preflight-v1";

constexpr int kTopK = 2;
constexpr int kExperts = 8;
constexpr int kCriticalToken = 0;
constexpr int kCriticalExpertA = 0;
constexpr int kCriticalExpertB = 1;
constexpr int kExpectedCriticalContributions = 2;
constexpr int kHidden = 256;
constexpr int kOutput = 1024;
constexpr int kRowsPerWarp = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerTile = kRowsPerWarp * kWarpsPerBlock;
constexpr int kProducerThreads = 32 * kWarpsPerBlock;
constexpr int kConsumerThreads = 256;
constexpr int kProjectionOutputs = 64;
constexpr int kMediumTokens = 4096;
constexpr int kHighTokens = 64000;
constexpr int kCalibrationWarmups = 10;
constexpr int kCalibrationRepeats = 50;
constexpr int kFormalWarmups = 30;
constexpr int kCorrectnessRepeats = 30;
constexpr int kUtilityRepeats = 200;
constexpr std::array<double, 4> kGateCandidates{{0.0, 0.125, 0.25, 0.5}};

enum class Variant : int {
  kAllDoneSham = 0,
  kEager = 1,
  kProgressGated = 2,
};

enum class WorkMode : int { kCorrectness = 0, kUtility = 1 };

enum class Phase : int { kPreflight = 0, kCalibration = 1, kFormal = 2 };

const char* VariantName(Variant value) {
  switch (value) {
    case Variant::kAllDoneSham:
      return "A_ALL_DONE_SHAM";
    case Variant::kEager:
      return "B_EAGER_JOINSTREAM";
    case Variant::kProgressGated:
      return "C_PROGRESS_GATED_JOINSTREAM";
  }
  return "UNKNOWN";
}

const char* ModeName(WorkMode value) {
  return value == WorkMode::kCorrectness ? "correctness" : "utility";
}

struct TileTask {
  int32_t expert = 0;
  int32_t routed_begin = 0;
  int32_t routed_count = 0;
  int32_t reserved = 0;
};

struct alignas(128) Control {
  alignas(4) uint32_t next_task;
  alignas(4) uint32_t gate_recorded;
  alignas(4) uint32_t critical_contributions_done;
  alignas(4) uint32_t join_ready;
  alignas(4) uint32_t producer_work_done;
  alignas(4) uint32_t producer_done;
  alignas(4) uint32_t consumer_entered;
  alignas(4) uint32_t stale_read;

  alignas(8) uint64_t producer_start_ns;
  uint64_t join_close_ns;
  uint64_t gate_satisfied_ns;
  uint64_t consumer_entry_ns;
  uint64_t consumer_observe_ns;
  uint64_t consumer_start_ns;
  uint64_t consumer_end_ns;
  uint64_t producer_end_ns;
  uint64_t output_hash;
  uint64_t critical_row_hash;
  uint32_t remaining_work_at_consumer_start;
  uint32_t reserved1;
};

struct RoutePlan {
  std::string cell_id;
  std::string distribution;
  std::string scale;
  uint64_t seed = 0;
  int total_tokens = 0;
  std::vector<int32_t> routed_token_ids;
  std::vector<int32_t> expert_offsets;
  std::vector<int32_t> expert_routed_token_counts;
  std::vector<TileTask> tasks;
  uint64_t route_hash = 0;
  uint64_t expert_token_counts_hash = 0;
  uint64_t theoretical_flops = 0;
  uint64_t producer_work_hash = 0;
};

struct LockCell {
  std::string cell_id;
  std::string distribution;
  std::string scale;
  std::string status = "SELECTED";
  double gate_ratio = 0.0;
  int total_tokens = 0;
  int routed_tokens = 0;
  std::vector<int32_t> expert_routed_token_counts;
  uint64_t theoretical_flops = 0;
};

struct RunLock {
  uint64_t calibration_seed = 0;
  uint64_t formal_seed = 0;
  std::vector<LockCell> cells;
};

struct Options {
  int device = 0;
  Phase phase = Phase::kCalibration;
  bool phase_set = false;
  uint64_t seed = 0;
  bool seed_set = false;
  uint64_t formal_seed = 0;
  bool formal_seed_set = false;
  std::string preflight_lock_path;
  std::string run_lock_path;
  std::string output_prefix = "joinstream_real_moe_tail";
};

struct DeviceBuffers {
  Control* control = nullptr;
  int32_t* routed_token_ids = nullptr;
  TileTask* tasks = nullptr;
  half* inputs = nullptr;
  half* weights = nullptr;
  float* routed_outputs = nullptr;
  float* critical_parts = nullptr;
  float* critical_row = nullptr;
  float* projection = nullptr;
  uint64_t* timer_probe = nullptr;
};

struct Streams {
  cudaStream_t producer = nullptr;
  cudaStream_t consumer = nullptr;
  cudaStream_t copy = nullptr;
  uint32_t* pinned_entered = nullptr;
};

struct TrialRecord {
  std::string phase;
  std::string mode;
  std::string repeat_kind;
  int repeat_index = 0;
  int permutation_slot = 0;
  std::string sample_order;
  Variant variant = Variant::kAllDoneSham;
  std::string cell_id;
  std::string route_distribution;
  std::string problem_scale;
  uint64_t route_seed = 0;
  std::string gate_selection_status;
  double gate_remaining_ratio = 0.0;
  uint32_t gate_threshold_remaining_blocks = 0;

  uint64_t route_table_hash = 0;
  uint64_t expert_token_counts_hash = 0;
  uint64_t input_hash = 0;
  int total_tokens = 0;
  int routed_tokens = 0;
  int top_k = kTopK;
  int expert_count = kExperts;
  uint64_t theoretical_flops = 0;
  uint64_t producer_work_hash = 0;
  uint64_t consumer_work_hash = 0;
  uint64_t progress_instrumentation_hash = 0;
  int producer_launches = 1;
  int consumer_launches = 1;
  int producer_grid_size = 0;
  int producer_block_size = kProducerThreads;
  int consumer_grid_size = 1;
  int consumer_block_size = kConsumerThreads;
  int expert_tiles_total = 0;
  bool synthetic_delay_enabled = false;
  bool artificial_sm_reservation_enabled = false;

  int critical_expert_a = kCriticalExpertA;
  int critical_expert_b = kCriticalExpertB;
  uint32_t critical_contributions_done = 0;
  uint32_t critical_contributions_expected =
      kExpectedCriticalContributions;
  uint32_t producer_work_units_done = 0;
  uint32_t producer_work_units_expected = 0;
  uint64_t output_hash = 0;
  uint64_t reference_output_hash = 0;
  bool correctness_pass = false;
  bool stale_read = false;
  bool timestamp_contract_pass = false;
  std::string cuda_error;

  uint64_t producer_start_ns = 0;
  uint64_t join_close_ns = 0;
  uint64_t gate_satisfied_ns = 0;
  uint64_t consumer_entry_ns = 0;
  uint64_t consumer_observe_ns = 0;
  uint64_t consumer_start_ns = 0;
  uint64_t consumer_end_ns = 0;
  uint64_t producer_end_ns = 0;
  uint64_t total_end_ns = 0;
  uint32_t remaining_producer_work_at_consumer_start = 0;
  int64_t join_to_gate_latency_ns = 0;
  int64_t notification_latency_ns = 0;
  int64_t natural_overlap_window_ns = 0;
};

#define CUDA_THROW(call)                                                       \
  do {                                                                         \
    const cudaError_t gate_cuda_error = (call);                                \
    if (gate_cuda_error != cudaSuccess) {                                      \
      std::ostringstream gate_cuda_message;                                    \
      gate_cuda_message << #call << ": " << cudaGetErrorString(gate_cuda_error);\
      throw std::runtime_error(gate_cuda_message.str());                       \
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

__global__ void TimerProbeKernel(uint64_t* minimum_positive_delta) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint64_t best = ~uint64_t{0};
  uint64_t previous = GlobalTimerNs();
  for (int index = 0; index < 4096; ++index) {
    const uint64_t now = GlobalTimerNs();
    if (now > previous) best = min(best, now - previous);
    previous = now;
  }
  *minimum_positive_delta = best == ~uint64_t{0} ? 0 : best;
}

uint64_t FnvBytes(const void* data, size_t bytes,
                  uint64_t hash = 1469598103934665603ull) {
  const auto* raw = static_cast<const unsigned char*>(data);
  for (size_t index = 0; index < bytes; ++index) {
    hash ^= raw[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t FnvString(const std::string& value) {
  return FnvBytes(value.data(), value.size());
}

uint64_t SplitMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char byte : value) {
    if (byte == '\\') out << "\\\\";
    else if (byte == '"') out << "\\\"";
    else if (byte == '\n') out << "\\n";
    else if (byte == '\r') out << "\\r";
    else if (byte == '\t') out << "\\t";
    else out << static_cast<char>(byte);
  }
  return out.str();
}

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string out = "\"";
  for (char byte : value) {
    if (byte == '"') out.push_back('"');
    out.push_back(byte);
  }
  out.push_back('"');
  return out;
}

int64_t SignedDifference(uint64_t lhs, uint64_t rhs) {
  if (lhs >= rhs) return static_cast<int64_t>(lhs - rhs);
  return -static_cast<int64_t>(rhs - lhs);
}

double Median(std::vector<double> values) {
  if (values.empty()) throw std::runtime_error("median of empty vector");
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : 0.5 * (values[middle - 1] + values[middle]);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto need = [&](const char* name) -> std::string {
      if (index + 1 >= argc) throw std::invalid_argument(std::string("missing ") + name);
      return argv[++index];
    };
    if (argument == "--phase") {
      const std::string value = need("--phase value");
      if (value == "preflight") options.phase = Phase::kPreflight;
      else if (value == "calibration") options.phase = Phase::kCalibration;
      else if (value == "formal") options.phase = Phase::kFormal;
      else throw std::invalid_argument(
          "--phase must be preflight, calibration, or formal");
      options.phase_set = true;
    } else if (argument == "--seed") {
      options.seed = std::stoull(need("--seed value"));
      options.seed_set = true;
    } else if (argument == "--formal-seed") {
      options.formal_seed = std::stoull(need("--formal-seed value"));
      options.formal_seed_set = true;
    } else if (argument == "--run-lock") {
      options.run_lock_path = need("--run-lock value");
    } else if (argument == "--preflight-lock") {
      options.preflight_lock_path = need("--preflight-lock value");
    } else if (argument == "--output-prefix") {
      options.output_prefix = need("--output-prefix value");
    } else if (argument == "--device") {
      options.device = std::stoi(need("--device value"));
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --phase preflight|calibration|formal --seed N "
                   "[--formal-seed N] [--preflight-lock PATH] "
                   "[--run-lock PATH] "
                   "[--device N] [--output-prefix PATH]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (!options.phase_set || !options.seed_set) {
    throw std::invalid_argument("--phase and --seed are required");
  }
  if (options.phase == Phase::kCalibration) {
    if (!options.formal_seed_set || options.formal_seed == options.seed) {
      throw std::invalid_argument(
          "calibration requires a distinct --formal-seed");
    }
    if (options.preflight_lock_path.empty()) {
      throw std::invalid_argument(
          "calibration requires --preflight-lock from a passing preflight");
    }
  } else if (options.phase == Phase::kFormal && options.run_lock_path.empty()) {
    throw std::invalid_argument("formal phase requires --run-lock");
  }
  return options;
}

RoutePlan BuildRoutePlan(const std::string& distribution,
                         const std::string& scale, uint64_t seed) {
  RoutePlan plan;
  plan.distribution = distribution;
  plan.scale = scale;
  plan.seed = seed;
  plan.total_tokens = scale == "MEDIUM_RESIDENCY" ? kMediumTokens : kHighTokens;
  plan.cell_id = distribution + "__" + scale;
  std::array<std::vector<int32_t>, kExperts> grouped;
  constexpr std::array<int, kExperts> skew_cdf{{40, 60, 72, 82, 89, 94, 98, 100}};

  // The seed permutes token identities, not the number of rows assigned to
  // each expert.  Calibration and formal therefore have distinct route tables
  // but exactly the same grouped-expert shape and FLOP count.
  std::vector<int> seeded_token_order;
  seeded_token_order.reserve(plan.total_tokens - 1);
  for (int token = 1; token < plan.total_tokens; ++token)
    seeded_token_order.push_back(token);
  std::stable_sort(seeded_token_order.begin(), seeded_token_order.end(),
                   [&](int lhs, int rhs) {
                     return SplitMix64(seed ^ static_cast<uint64_t>(lhs)) <
                            SplitMix64(seed ^ static_cast<uint64_t>(rhs));
                   });

  for (int rank = 0; rank < plan.total_tokens; ++rank) {
    const int token = rank == 0 ? kCriticalToken : seeded_token_order[rank - 1];
    int first = 0;
    int second = 1;
    if (token != kCriticalToken) {
      if (distribution == "BALANCED") {
        first = (rank * 2) % kExperts;
        second = (rank * 2 + 1) % kExperts;
      } else {
        const int draw = rank % 100;
        first = static_cast<int>(std::lower_bound(
                    skew_cdf.begin(), skew_cdf.end(), draw + 1) -
                skew_cdf.begin());
        second = (first + 1) % kExperts;
      }
    }
    grouped[first].push_back(token);
    grouped[second].push_back(token);
  }

  // Stable seed-dependent order, with the critical row frozen first in its
  // two distinct experts so the join is a real K=2 data dependency.
  for (int expert = 0; expert < kExperts; ++expert) {
    auto& rows = grouped[expert];
    std::stable_sort(rows.begin(), rows.end(), [&](int32_t lhs, int32_t rhs) {
      if (lhs == kCriticalToken || rhs == kCriticalToken)
        return lhs == kCriticalToken;
      return SplitMix64(seed ^ (static_cast<uint64_t>(expert) << 48) ^ lhs) <
             SplitMix64(seed ^ (static_cast<uint64_t>(expert) << 48) ^ rhs);
    });
  }

  plan.expert_offsets.push_back(0);
  for (int expert = 0; expert < kExperts; ++expert) {
    plan.expert_routed_token_counts.push_back(
        static_cast<int32_t>(grouped[expert].size()));
    plan.routed_token_ids.insert(plan.routed_token_ids.end(),
                                 grouped[expert].begin(), grouped[expert].end());
    plan.expert_offsets.push_back(
        static_cast<int32_t>(plan.routed_token_ids.size()));
  }
  if (static_cast<int>(plan.routed_token_ids.size()) !=
      plan.total_tokens * kTopK) {
    throw std::runtime_error("route table lost routed tokens");
  }

  std::vector<TileTask> expert_major;
  std::vector<TileTask> critical;
  for (int expert = 0; expert < kExperts; ++expert) {
    const int begin = plan.expert_offsets[expert];
    const int end = plan.expert_offsets[expert + 1];
    for (int offset = begin; offset < end; offset += kRowsPerTile) {
      TileTask task{expert, offset, std::min(kRowsPerTile, end - offset), 0};
      const bool contains_critical =
          expert <= kCriticalExpertB && offset == begin &&
          plan.routed_token_ids[offset] == kCriticalToken;
      expert_major.push_back(task);
      if (contains_critical) critical.push_back(task);
    }
  }
  if (critical.size() != 2 || critical[0].expert == critical[1].expert) {
    throw std::runtime_error("critical K=2 tasks are not distinct experts");
  }
  // Standard fixed expert-major grouped dispatch.  The critical row remains
  // first inside each of experts 0 and 1, but expert 1's K=2 contribution is
  // issued only at its natural expert-major position after expert 0's tiles.
  // Therefore any residual tail after the join comes from route-induced group
  // sizes, never from moving critical work forward or injecting delay work.
  plan.tasks = std::move(expert_major);

  plan.route_hash = FnvBytes(plan.routed_token_ids.data(),
                            plan.routed_token_ids.size() * sizeof(int32_t));
  plan.route_hash = FnvBytes(plan.expert_offsets.data(),
                            plan.expert_offsets.size() * sizeof(int32_t),
                            plan.route_hash);
  plan.expert_token_counts_hash = FnvBytes(
      plan.expert_routed_token_counts.data(),
      plan.expert_routed_token_counts.size() * sizeof(int32_t));
  plan.theoretical_flops = 2ull * plan.routed_token_ids.size() * kHidden * kOutput;
  std::ostringstream contract;
  contract << "fp16_grouped_expert_wmma_v1;H=" << kHidden << ";N=" << kOutput
           << ";rows=" << plan.routed_token_ids.size() << ";tiles="
           << plan.tasks.size() << ";route=" << plan.route_hash;
  plan.producer_work_hash = FnvString(contract.str());
  return plan;
}

std::vector<RoutePlan> BuildFourPlans(uint64_t seed) {
  std::vector<RoutePlan> plans;
  for (const char* distribution : {"BALANCED", "SKEWED"}) {
    for (const char* scale : {"MEDIUM_RESIDENCY", "HIGH_RESIDENCY"}) {
      plans.push_back(BuildRoutePlan(distribution, scale, seed));
    }
  }
  return plans;
}

__device__ __forceinline__ uint64_t DeviceHashRow(const float* row) {
  uint64_t hash = 1469598103934665603ull;
  float sum = 0.0f;
  for (int index = 0; index < kOutput; ++index) {
    hash = FnvMix(hash, __float_as_uint(row[index]));
    sum += row[index];
  }
  return FnvMix(hash, __float_as_uint(sum));
}

__global__ void GroupedExpertProducer(
    Control* control, const int32_t* routed_token_ids,
    const TileTask* tasks, int task_count, const half* inputs,
    const half* weights, float* routed_outputs, float* critical_parts,
    float* critical_row, uint32_t gate_remaining_blocks) {
  using namespace nvcuda;
  __shared__ uint32_t task_index;
  __shared__ __align__(32) half a_tiles[kWarpsPerBlock][16 * 16];
  __shared__ __align__(32) float c_tiles[kWarpsPerBlock][16 * 16];

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> next_task(
      control->next_task);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> gate_recorded(
      control->gate_recorded);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> critical_done(
      control->critical_contributions_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> join_ready(
      control->join_ready);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> work_done(
      control->producer_work_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> producer_done(
      control->producer_done);

  if (threadIdx.x == 0) {
    task_index = next_task.fetch_add(1u, cuda::memory_order_relaxed);
    atomicMin(reinterpret_cast<unsigned long long*>(&control->producer_start_ns),
              static_cast<unsigned long long>(GlobalTimerNs()));
  }
  __syncthreads();
  if (task_index >= static_cast<uint32_t>(task_count)) return;

  const TileTask task = tasks[task_index];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int warp_row_begin = warp * kRowsPerWarp;
  const int valid_rows = max(0, min(kRowsPerWarp,
                                    task.routed_count - warp_row_begin));

  for (int output_tile = 0; output_tile < kOutput / 16; ++output_tile) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int hidden_tile = 0; hidden_tile < kHidden / 16; ++hidden_tile) {
      for (int linear = lane; linear < 16 * 16; linear += 32) {
        const int local_row = linear / 16;
        const int hidden_column = linear % 16;
        half value = __float2half(0.0f);
        if (local_row < valid_rows) {
          const int routed_index =
              task.routed_begin + warp_row_begin + local_row;
          const int token = routed_token_ids[routed_index];
          value = inputs[static_cast<size_t>(token) * kHidden +
                         hidden_tile * 16 + hidden_column];
        }
        a_tiles[warp][linear] = value;
      }
      __syncwarp();

      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                     wmma::row_major>
          a_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                     wmma::col_major>
          b_fragment;
      wmma::load_matrix_sync(a_fragment, a_tiles[warp], 16);
      const half* weight_tile =
          weights + static_cast<size_t>(task.expert) * kHidden * kOutput +
          static_cast<size_t>(output_tile) * 16 * kHidden + hidden_tile * 16;
      wmma::load_matrix_sync(b_fragment, weight_tile, kHidden);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    wmma::store_matrix_sync(c_tiles[warp], accumulator, 16,
                            wmma::mem_row_major);
    __syncwarp();
    for (int linear = lane; linear < 16 * 16; linear += 32) {
      const int local_row = linear / 16;
      const int output_column = linear % 16;
      if (local_row < valid_rows) {
        const int routed_index = task.routed_begin + warp_row_begin + local_row;
        routed_outputs[static_cast<size_t>(routed_index) * kOutput +
                       output_tile * 16 + output_column] = c_tiles[warp][linear];
      }
    }
    __syncwarp();
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int critical_routed_index = -1;
    if (task.expert == kCriticalExpertA || task.expert == kCriticalExpertB) {
      for (int row = 0; row < task.routed_count; ++row) {
        const int routed_index = task.routed_begin + row;
        if (routed_token_ids[routed_index] == kCriticalToken) {
          critical_routed_index = routed_index;
          break;
        }
      }
    }
    if (critical_routed_index >= 0) {
      const int slot = task.expert == kCriticalExpertA ? 0 : 1;
      for (int column = 0; column < kOutput; ++column) {
        critical_parts[static_cast<size_t>(slot) * kOutput + column] =
            routed_outputs[static_cast<size_t>(critical_routed_index) *
                               kOutput +
                           column];
      }
      const uint32_t prior =
          critical_done.fetch_add(1u, cuda::memory_order_acq_rel);
      if (prior == kExpectedCriticalContributions - 1) {
        for (int column = 0; column < kOutput; ++column) {
          critical_row[column] = critical_parts[column] +
                                 critical_parts[kOutput + column];
        }
        control->join_close_ns = GlobalTimerNs();
        join_ready.store(1u, cuda::memory_order_release);
      }
    }

    const uint32_t prior = work_done.fetch_add(1u, cuda::memory_order_acq_rel);
    const uint32_t remaining =
        static_cast<uint32_t>(task_count) - (prior + 1u);
    if (remaining <= gate_remaining_blocks) {
      uint32_t expected = 0u;
      if (gate_recorded.compare_exchange_strong(
              expected, 1u, cuda::memory_order_acq_rel,
              cuda::memory_order_acquire)) {
        control->gate_satisfied_ns = GlobalTimerNs();
        // 0=unset, 1=claimed, 2=timestamp published.  The release prevents C
        // from observing a satisfied gate with a stale zero timestamp.
        gate_recorded.store(2u, cuda::memory_order_release);
      }
    }
    if (prior == static_cast<uint32_t>(task_count - 1)) {
      control->producer_end_ns = GlobalTimerNs();
      producer_done.store(1u, cuda::memory_order_release);
    }
  }
}

__global__ void CriticalConsumer(Control* control, const float* critical_row,
                                 float* projection, int variant_value,
                                 uint32_t total_work,
                                 uint32_t gate_remaining_blocks,
                                 int mode_value) {
  __shared__ float reduction[kConsumerThreads];
  __shared__ float outputs[kProjectionOutputs];
  __shared__ float inverse_rms;

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> join_ready(
      control->join_ready);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> work_done(
      control->producer_work_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> producer_done(
      control->producer_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> gate_recorded(
      control->gate_recorded);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> consumer_entered(
      control->consumer_entered);

  if (threadIdx.x == 0) {
    control->consumer_entry_ns = GlobalTimerNs();
    __threadfence_system();
    consumer_entered.store(1u, cuda::memory_order_release);
    __threadfence_system();

    uint32_t observed_work = 0u;
    while (true) {
      // Exactly the same join/progress/done atomic loads occur in every
      // variant.  Only this final readiness predicate differs.
      const uint32_t observed_join =
          join_ready.load(cuda::memory_order_acquire);
      observed_work = work_done.load(cuda::memory_order_acquire);
      const uint32_t observed_done =
          producer_done.load(cuda::memory_order_acquire);
      const uint32_t observed_gate =
          gate_recorded.load(cuda::memory_order_acquire);
      const bool ready =
          variant_value == static_cast<int>(Variant::kAllDoneSham)
              ? (observed_join != 0u && observed_done != 0u)
          : variant_value == static_cast<int>(Variant::kEager)
              ? (observed_join != 0u)
              : (observed_join != 0u && observed_gate == 2u);
      if (ready) break;
    }
    control->consumer_observe_ns = GlobalTimerNs();
    // A common post-predicate progress load also ensures C reads progress
    // after acquiring gate state 2 (the producer's publication release).
    observed_work = work_done.load(cuda::memory_order_acquire);
    control->remaining_work_at_consumer_start = total_work - observed_work;
  }
  (void)gate_remaining_blocks;
  __syncthreads();
  (void)join_ready.load(cuda::memory_order_acquire);
  __syncthreads();

  if (threadIdx.x == 0) control->consumer_start_ns = GlobalTimerNs();
  // No barrier exists between this conservative timestamp and useful work.
  uint64_t row_hash = 0;
  if (threadIdx.x == 0) row_hash = DeviceHashRow(critical_row);

  uint64_t work_hash = 0;
  if (mode_value == static_cast<int>(WorkMode::kCorrectness)) {
    if (threadIdx.x == 0) work_hash = row_hash;
  } else {
    float local_sum = 0.0f;
    for (int column = threadIdx.x; column < kOutput; column += blockDim.x) {
      const float value = critical_row[column];
      local_sum = fmaf(value, value, local_sum);
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride)
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      inverse_rms = rsqrtf(reduction[0] / static_cast<float>(kOutput) + 1e-5f);
    }
    __syncthreads();
    if (threadIdx.x < kProjectionOutputs) {
      float value = 0.0f;
      for (int column = 0; column < kOutput; ++column) {
        const int weight =
            (threadIdx.x * 131 + column * 17) % 257 - 128;
        value = fmaf(critical_row[column] * inverse_rms,
                     static_cast<float>(weight) * (1.0f / 4096.0f), value);
      }
      outputs[threadIdx.x] = value;
      projection[threadIdx.x] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      work_hash = row_hash;
      int top_one = 0;
      int top_two = 1;
      for (int output = 0; output < kProjectionOutputs; ++output) {
        work_hash = FnvMix(work_hash, __float_as_uint(outputs[output]));
        if (outputs[output] > outputs[top_one]) {
          top_two = top_one;
          top_one = output;
        } else if (output != top_one && outputs[output] > outputs[top_two]) {
          top_two = output;
        }
      }
      work_hash = FnvMix(work_hash, static_cast<uint32_t>(top_one));
      work_hash = FnvMix(work_hash, static_cast<uint32_t>(top_two));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    control->critical_row_hash = row_hash;
    control->output_hash = work_hash;
    control->consumer_end_ns = GlobalTimerNs();
  }
}

void AllocateBuffers(DeviceBuffers* buffers, int maximum_tokens,
                     int maximum_routed_tokens, int maximum_tasks) {
  CUDA_THROW(cudaMalloc(&buffers->control, sizeof(Control)));
  CUDA_THROW(cudaMalloc(&buffers->routed_token_ids,
                        sizeof(int32_t) * maximum_routed_tokens));
  CUDA_THROW(cudaMalloc(&buffers->tasks, sizeof(TileTask) * maximum_tasks));
  CUDA_THROW(cudaMalloc(&buffers->inputs,
                        sizeof(half) * maximum_tokens * kHidden));
  CUDA_THROW(cudaMalloc(&buffers->weights,
                        sizeof(half) * kExperts * kHidden * kOutput));
  CUDA_THROW(cudaMalloc(&buffers->routed_outputs,
                        sizeof(float) * maximum_routed_tokens * kOutput));
  CUDA_THROW(cudaMalloc(&buffers->critical_parts,
                        sizeof(float) * kTopK * kOutput));
  CUDA_THROW(cudaMalloc(&buffers->critical_row, sizeof(float) * kOutput));
  CUDA_THROW(cudaMalloc(&buffers->projection,
                        sizeof(float) * kProjectionOutputs));
  CUDA_THROW(cudaMalloc(&buffers->timer_probe, sizeof(uint64_t)));
}

void FreeBuffers(DeviceBuffers* buffers) {
  cudaFree(buffers->timer_probe);
  cudaFree(buffers->projection);
  cudaFree(buffers->critical_row);
  cudaFree(buffers->critical_parts);
  cudaFree(buffers->routed_outputs);
  cudaFree(buffers->weights);
  cudaFree(buffers->inputs);
  cudaFree(buffers->tasks);
  cudaFree(buffers->routed_token_ids);
  cudaFree(buffers->control);
  *buffers = DeviceBuffers{};
}

uint64_t MeasureTimerResolution(uint64_t* timer_probe) {
  CUDA_THROW(cudaMemset(timer_probe, 0, sizeof(uint64_t)));
  TimerProbeKernel<<<1, 1>>>(timer_probe);
  CUDA_THROW(cudaGetLastError());
  CUDA_THROW(cudaDeviceSynchronize());
  uint64_t resolution = 0;
  CUDA_THROW(cudaMemcpy(&resolution, timer_probe, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  return resolution;
}

uint64_t UploadFixedInputs(DeviceBuffers* buffers, int maximum_tokens,
                           uint64_t seed) {
  std::vector<half> inputs(static_cast<size_t>(maximum_tokens) * kHidden);
  std::vector<half> weights(static_cast<size_t>(kExperts) * kHidden * kOutput);
  for (int token = 0; token < maximum_tokens; ++token) {
    for (int hidden = 0; hidden < kHidden; ++hidden) {
      const uint64_t bits = SplitMix64(seed ^
                                      (static_cast<uint64_t>(token) << 20) ^
                                      static_cast<uint64_t>(hidden));
      const float value = static_cast<float>(static_cast<int>(bits % 257) - 128) /
                          128.0f;
      inputs[static_cast<size_t>(token) * kHidden + hidden] =
          __float2half(value);
    }
  }
  // Column-major [N, K] per expert, matching WMMA matrix_b col_major.
  for (int expert = 0; expert < kExperts; ++expert) {
    for (int output = 0; output < kOutput; ++output) {
      for (int hidden = 0; hidden < kHidden; ++hidden) {
        const uint64_t bits = SplitMix64(
            seed ^ (static_cast<uint64_t>(expert) << 56) ^
            (static_cast<uint64_t>(output) << 20) ^ hidden);
        const float value =
            static_cast<float>(static_cast<int>(bits % 129) - 64) / 256.0f;
        weights[static_cast<size_t>(expert) * kHidden * kOutput +
                static_cast<size_t>(output) * kHidden + hidden] =
            __float2half(value);
      }
    }
  }
  uint64_t hash = FnvBytes(inputs.data(), inputs.size() * sizeof(half));
  hash = FnvBytes(weights.data(), weights.size() * sizeof(half), hash);
  CUDA_THROW(cudaMemcpy(buffers->inputs, inputs.data(),
                        inputs.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_THROW(cudaMemcpy(buffers->weights, weights.data(),
                        weights.size() * sizeof(half), cudaMemcpyHostToDevice));
  return hash;
}

void UploadPlan(const DeviceBuffers& buffers, const RoutePlan& plan) {
  CUDA_THROW(cudaMemcpy(buffers.routed_token_ids, plan.routed_token_ids.data(),
                        plan.routed_token_ids.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_THROW(cudaMemcpy(buffers.tasks, plan.tasks.data(),
                        plan.tasks.size() * sizeof(TileTask),
                        cudaMemcpyHostToDevice));
}

void ResetTrial(const DeviceBuffers& buffers, const RoutePlan& plan) {
  Control initial{};
  initial.producer_start_ns = std::numeric_limits<uint64_t>::max();
  CUDA_THROW(cudaMemcpy(buffers.control, &initial, sizeof(Control),
                        cudaMemcpyHostToDevice));
  CUDA_THROW(cudaMemset(buffers.critical_parts, 0,
                        sizeof(float) * kTopK * kOutput));
  CUDA_THROW(cudaMemset(buffers.critical_row, 0, sizeof(float) * kOutput));
  CUDA_THROW(cudaMemset(buffers.projection, 0,
                        sizeof(float) * kProjectionOutputs));
  CUDA_THROW(cudaDeviceSynchronize());
}

void CreateStreams(Streams* streams) {
  CUDA_THROW(cudaStreamCreateWithFlags(&streams->producer,
                                       cudaStreamNonBlocking));
  CUDA_THROW(cudaStreamCreateWithFlags(&streams->consumer,
                                       cudaStreamNonBlocking));
  CUDA_THROW(cudaStreamCreateWithFlags(&streams->copy, cudaStreamNonBlocking));
  CUDA_THROW(cudaHostAlloc(&streams->pinned_entered, sizeof(uint32_t),
                           cudaHostAllocDefault));
}

void DestroyStreams(Streams* streams) {
  cudaFreeHost(streams->pinned_entered);
  cudaStreamDestroy(streams->copy);
  cudaStreamDestroy(streams->consumer);
  cudaStreamDestroy(streams->producer);
  *streams = Streams{};
}

void WaitForConsumerEntry(const DeviceBuffers& buffers,
                          const Streams& streams) {
  *streams.pinned_entered = 0u;
  const auto* base = reinterpret_cast<const unsigned char*>(buffers.control);
  const void* address = base + offsetof(Control, consumer_entered);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  while (*streams.pinned_entered == 0u) {
    CUDA_THROW(cudaMemcpyAsync(streams.pinned_entered, address,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               streams.copy));
    CUDA_THROW(cudaStreamSynchronize(streams.copy));
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("consumer entry handshake timed out");
    }
  }
}

bool TimestampContract(const TrialRecord& row) {
  if (row.consumer_entry_ns > row.producer_start_ns ||
      row.producer_start_ns == 0 ||
      row.producer_start_ns == std::numeric_limits<uint64_t>::max() ||
      row.join_close_ns < row.producer_start_ns ||
      row.consumer_start_ns < row.consumer_observe_ns ||
      row.consumer_end_ns < row.consumer_start_ns ||
      row.producer_end_ns < row.join_close_ns ||
      row.gate_satisfied_ns < row.producer_start_ns ||
      row.gate_satisfied_ns > row.producer_end_ns ||
      row.total_end_ns < std::max(row.producer_end_ns, row.consumer_end_ns)) {
    return false;
  }
  if (row.variant == Variant::kAllDoneSham &&
      (row.remaining_producer_work_at_consumer_start != 0 ||
       row.consumer_observe_ns < row.producer_end_ns)) {
    return false;
  }
  if (row.variant == Variant::kEager &&
      row.consumer_observe_ns < row.join_close_ns) return false;
  if (row.variant == Variant::kProgressGated) {
    const uint32_t allowed = static_cast<uint32_t>(std::ceil(
        row.expert_tiles_total * row.gate_remaining_ratio));
    if (row.remaining_producer_work_at_consumer_start > allowed ||
        row.consumer_observe_ns <
            std::max(row.join_close_ns, row.gate_satisfied_ns))
      return false;
  }
  return true;
}

TrialRecord RunTrial(const DeviceBuffers& buffers, const Streams& streams,
                     const RoutePlan& plan, uint64_t input_hash,
                     uint64_t route_seed, Phase phase, WorkMode mode,
                     const std::string& repeat_kind, int repeat_index,
                     int permutation_slot, const std::string& sample_order,
                     Variant variant, double gate_ratio,
                     const std::string& gate_selection_status) {
  ResetTrial(buffers, plan);
  const uint32_t total_work = static_cast<uint32_t>(plan.tasks.size());
  const uint32_t gate_remaining = static_cast<uint32_t>(
      std::ceil(static_cast<double>(total_work) * gate_ratio));

  CriticalConsumer<<<1, kConsumerThreads, 0, streams.consumer>>>(
      buffers.control, buffers.critical_row, buffers.projection,
      static_cast<int>(variant), total_work, gate_remaining,
      static_cast<int>(mode));
  CUDA_THROW(cudaGetLastError());
  WaitForConsumerEntry(buffers, streams);
  GroupedExpertProducer<<<static_cast<int>(plan.tasks.size()),
                          kProducerThreads, 0, streams.producer>>>(
      buffers.control, buffers.routed_token_ids, buffers.tasks,
      static_cast<int>(plan.tasks.size()), buffers.inputs, buffers.weights,
      buffers.routed_outputs, buffers.critical_parts, buffers.critical_row,
      gate_remaining);
  CUDA_THROW(cudaGetLastError());
  CUDA_THROW(cudaDeviceSynchronize());

  Control control{};
  CUDA_THROW(cudaMemcpy(&control, buffers.control, sizeof(Control),
                        cudaMemcpyDeviceToHost));
  TrialRecord row;
  row.phase = phase == Phase::kCalibration ? "calibration" : "formal";
  row.mode = ModeName(mode);
  row.repeat_kind = repeat_kind;
  row.repeat_index = repeat_index;
  row.permutation_slot = permutation_slot;
  row.sample_order = sample_order;
  row.variant = variant;
  row.cell_id = plan.cell_id;
  row.route_distribution = plan.distribution;
  row.problem_scale = plan.scale;
  row.route_seed = route_seed;
  row.gate_selection_status = gate_selection_status;
  row.gate_remaining_ratio = gate_ratio;
  row.gate_threshold_remaining_blocks = gate_remaining;
  row.route_table_hash = plan.route_hash;
  row.expert_token_counts_hash = plan.expert_token_counts_hash;
  row.input_hash = input_hash;
  row.total_tokens = plan.total_tokens;
  row.routed_tokens = static_cast<int>(plan.routed_token_ids.size());
  row.theoretical_flops = plan.theoretical_flops;
  row.producer_work_hash = plan.producer_work_hash;
  row.consumer_work_hash = FnvString(
      mode == WorkMode::kCorrectness
          ? "critical_row_bit_hash_v1"
          : "critical_row_rmsnorm_projection64_top2_v1");
  row.progress_instrumentation_hash =
      FnvString("device_work_done_acq_rel_remaining_tiles_v1");
  row.producer_grid_size = static_cast<int>(plan.tasks.size());
  row.expert_tiles_total = static_cast<int>(plan.tasks.size());
  row.producer_work_units_expected = static_cast<uint32_t>(plan.tasks.size());
  row.producer_start_ns = control.producer_start_ns;
  row.join_close_ns = control.join_close_ns;
  row.gate_satisfied_ns = control.gate_satisfied_ns;
  row.consumer_entry_ns = control.consumer_entry_ns;
  row.consumer_observe_ns = control.consumer_observe_ns;
  row.consumer_start_ns = control.consumer_start_ns;
  row.consumer_end_ns = control.consumer_end_ns;
  row.producer_end_ns = control.producer_end_ns;
  row.total_end_ns = std::max(row.consumer_end_ns, row.producer_end_ns);
  row.remaining_producer_work_at_consumer_start =
      control.remaining_work_at_consumer_start;
  row.critical_contributions_done = control.critical_contributions_done;
  row.producer_work_units_done = control.producer_work_done;
  row.output_hash = control.output_hash;
  row.stale_read = control.stale_read != 0;
  row.join_to_gate_latency_ns =
      SignedDifference(row.gate_satisfied_ns, row.join_close_ns);
  const uint64_t activation_ns =
      variant == Variant::kAllDoneSham
          ? row.producer_end_ns
          : (variant == Variant::kEager
                 ? row.join_close_ns
                 : std::max(row.join_close_ns, row.gate_satisfied_ns));
  row.notification_latency_ns =
      SignedDifference(row.consumer_observe_ns, activation_ns);
  row.natural_overlap_window_ns =
      SignedDifference(row.producer_end_ns, row.consumer_start_ns);
  row.timestamp_contract_pass = TimestampContract(row);
  return row;
}

void WriteCsvHeader(std::ostream& out) {
  out << "schema_version,phase,mode,cell_id,route_distribution,problem_scale,"
         "route_seed,repeat_kind,repeat_index,permutation_slot,sample_order,"
         "variant,gate_selection_status,gate_remaining_ratio,"
         "gate_threshold_remaining_ratio,"
         "gate_threshold_remaining_blocks,route_table_hash,"
         "expert_token_counts_hash,input_hash,total_tokens,routed_tokens,top_k,"
         "expert_count,theoretical_flops,producer_work_hash,consumer_work_hash,"
         "progress_instrumentation_hash,producer_launches,consumer_launches,"
         "producer_grid_size,producer_block_size,consumer_grid_size,"
         "consumer_block_size,expert_tiles_total,synthetic_delay_enabled,"
         "artificial_sm_reservation_enabled,critical_expert_a,critical_expert_b,"
         "critical_contributions_done,critical_contributions_expected,"
         "producer_work_units_done,producer_work_units_expected,"
         "producer_work_done,producer_work_expected,output_hash,"
         "reference_output_hash,correctness_pass,stale_read,"
         "timestamp_contract_pass,cuda_error,producer_start_ns,join_close_ns,"
         "gate_satisfied_ns,consumer_entry_ns,consumer_observe_ns,"
         "consumer_start_ns,consumer_end_ns,producer_end_ns,total_end_ns,"
         "remaining_producer_work_at_consumer_start,join_to_gate_latency_ns,"
         "notification_latency_ns,natural_overlap_window_ns\n";
}

void WriteCsvRow(std::ostream& out, const TrialRecord& row) {
  out << kRawSchema << ',' << row.phase << ',' << row.mode << ','
      << CsvEscape(row.cell_id) << ',' << row.route_distribution << ','
      << row.problem_scale << ',' << row.route_seed << ',' << row.repeat_kind
      << ',' << row.repeat_index << ',' << row.permutation_slot << ','
      << row.sample_order << ',' << VariantName(row.variant) << ','
      << row.gate_selection_status << ',' << std::setprecision(8)
      << row.gate_remaining_ratio << ',' << row.gate_remaining_ratio << ','
      << row.gate_threshold_remaining_blocks << ',' << row.route_table_hash
      << ',' << row.expert_token_counts_hash << ',' << row.input_hash << ','
      << row.total_tokens << ',' << row.routed_tokens << ',' << row.top_k << ','
      << row.expert_count << ',' << row.theoretical_flops << ','
      << row.producer_work_hash << ',' << row.consumer_work_hash << ','
      << row.progress_instrumentation_hash << ',' << row.producer_launches << ','
      << row.consumer_launches << ',' << row.producer_grid_size << ','
      << row.producer_block_size << ',' << row.consumer_grid_size << ','
      << row.consumer_block_size << ',' << row.expert_tiles_total << ','
      << (row.synthetic_delay_enabled ? 1 : 0) << ','
      << (row.artificial_sm_reservation_enabled ? 1 : 0) << ','
      << row.critical_expert_a << ',' << row.critical_expert_b << ','
      << row.critical_contributions_done << ','
      << row.critical_contributions_expected << ','
      << row.producer_work_units_done << ','
      << row.producer_work_units_expected << ','
      << row.producer_work_units_done << ','
      << row.producer_work_units_expected << ',' << row.output_hash << ','
      << row.reference_output_hash << ',' << (row.correctness_pass ? 1 : 0)
      << ',' << (row.stale_read ? 1 : 0) << ','
      << (row.timestamp_contract_pass ? 1 : 0) << ','
      << CsvEscape(row.cuda_error) << ',' << row.producer_start_ns << ','
      << row.join_close_ns << ',' << row.gate_satisfied_ns << ','
      << row.consumer_entry_ns << ',' << row.consumer_observe_ns << ','
      << row.consumer_start_ns << ',' << row.consumer_end_ns << ','
      << row.producer_end_ns << ',' << row.total_end_ns << ','
      << row.remaining_producer_work_at_consumer_start << ','
      << row.join_to_gate_latency_ns << ',' << row.notification_latency_ns << ','
      << row.natural_overlap_window_ns << '\n';
}

void FinalizePairOrTriple(std::vector<TrialRecord*> rows) {
  if (rows.empty()) return;
  uint64_t reference = 0;
  for (TrialRecord* row : rows) {
    if (row->variant == Variant::kAllDoneSham) reference = row->output_hash;
  }
  for (TrialRecord* row : rows) {
    row->reference_output_hash = reference;
    row->correctness_pass =
        reference != 0 && row->output_hash == reference && !row->stale_read &&
        row->timestamp_contract_pass &&
        row->critical_contributions_done ==
            row->critical_contributions_expected &&
        row->producer_work_units_done == row->producer_work_units_expected;
  }
}

std::string CountsJson(const std::vector<int32_t>& counts) {
  std::ostringstream out;
  out << '[';
  for (size_t index = 0; index < counts.size(); ++index) {
    if (index) out << ',';
    out << counts[index];
  }
  out << ']';
  return out.str();
}

uint32_t RotateRight(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32u - bits));
}

// Small self-contained SHA-256 used only to bind the already-closed
// calibration JSON into the formal run lock.
std::string Sha256Hex(const std::string& bytes) {
  static constexpr std::array<uint32_t, 64> constants{{
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
      0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
      0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
      0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
      0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
      0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
      0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
      0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
      0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2}};
  std::vector<uint8_t> padded(bytes.begin(), bytes.end());
  const uint64_t bit_count = static_cast<uint64_t>(padded.size()) * 8ull;
  padded.push_back(0x80u);
  while (padded.size() % 64 != 56) padded.push_back(0u);
  for (int shift = 56; shift >= 0; shift -= 8)
    padded.push_back(static_cast<uint8_t>(bit_count >> shift));
  std::array<uint32_t, 8> hash{{0x6a09e667,0xbb67ae85,0x3c6ef372,
                                0xa54ff53a,0x510e527f,0x9b05688c,
                                0x1f83d9ab,0x5be0cd19}};
  for (size_t offset = 0; offset < padded.size(); offset += 64) {
    std::array<uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      const size_t at = offset + static_cast<size_t>(index) * 4;
      words[index] = (static_cast<uint32_t>(padded[at]) << 24) |
                     (static_cast<uint32_t>(padded[at + 1]) << 16) |
                     (static_cast<uint32_t>(padded[at + 2]) << 8) |
                     padded[at + 3];
    }
    for (int index = 16; index < 64; ++index) {
      const uint32_t s0 = RotateRight(words[index - 15], 7) ^
                          RotateRight(words[index - 15], 18) ^
                          (words[index - 15] >> 3);
      const uint32_t s1 = RotateRight(words[index - 2], 17) ^
                          RotateRight(words[index - 2], 19) ^
                          (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint32_t a=hash[0], b=hash[1], c=hash[2], d=hash[3];
    uint32_t e=hash[4], f=hash[5], g=hash[6], h=hash[7];
    for (int index = 0; index < 64; ++index) {
      const uint32_t big1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
                            RotateRight(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = h + big1 + choose + constants[index] + words[index];
      const uint32_t big0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
                            RotateRight(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = big0 + majority;
      h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    hash[0]+=a; hash[1]+=b; hash[2]+=c; hash[3]+=d;
    hash[4]+=e; hash[5]+=f; hash[6]+=g; hash[7]+=h;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint32_t word : hash) out << std::setw(8) << word;
  return out.str();
}

std::string CudaApiVersionString(int version) {
  std::ostringstream out;
  out << version / 1000 << '.' << (version % 1000) / 10;
  return out.str();
}

std::string NvidiaDriverDisplayVersion(int driver_api_version) {
  std::ifstream input("/proc/driver/nvidia/version");
  std::ostringstream bytes;
  bytes << input.rdbuf();
  const std::string text = bytes.str();
  for (size_t begin = 0; begin < text.size(); ++begin) {
    if (!std::isdigit(static_cast<unsigned char>(text[begin]))) continue;
    size_t end = begin;
    int dots = 0;
    while (end < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[end])) ||
            text[end] == '.')) {
      if (text[end] == '.') ++dots;
      ++end;
    }
    if (dots == 2) return text.substr(begin, end - begin);
    begin = end;
  }
  return CudaApiVersionString(driver_api_version);
}

void WriteEnvironment(const std::string& path, const cudaDeviceProp& property,
                      int driver_version, int runtime_version,
                      uint64_t timer_resolution_ns) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create environment JSON");
  out << "{\n  \"schema\": \"" << kEnvironmentSchema << "\",\n"
      << "  \"status\": \"CAPTURED\",\n"
      << "  \"gpu_available\": true,\n"
      << "  \"hardware\": {\"name\": \"" << JsonEscape(property.name)
      << "\", \"sm_count\": " << property.multiProcessorCount
      << ", \"compute_capability\": \"" << property.major << '.'
      << property.minor << "\"},\n"
      << "  \"software\": {\"driver_version\": \""
      << NvidiaDriverDisplayVersion(driver_version)
      << "\", \"cuda_toolkit_version\": \""
      << CudaApiVersionString(runtime_version)
      << "\", \"cuda_driver_api_version\": " << driver_version
      << ", \"cuda_runtime_api_version\": " << runtime_version << "},\n"
      << "  \"timer\": {\"source\": \"ptx_%globaltimer\", "
         "\"resolution_ns\": " << timer_resolution_ns << "}\n}\n";
}

void WriteBlocked(const std::string& prefix, const std::string& reason) {
  std::ofstream out(prefix + ".environment.json");
  out << "{\"schema\":\"" << kEnvironmentSchema
      << "\",\"status\":\"BLOCKED_NO_GPU\",\"gpu_available\":false,"
         "\"reason\":\""
      << JsonEscape(reason) << "\"}\n";
}

std::string ExtractString(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  size_t position = text.find(needle);
  if (position == std::string::npos) throw std::runtime_error("lock missing " + key);
  position = text.find(':', position + needle.size());
  position = text.find('"', position + 1);
  const size_t end = text.find('"', position + 1);
  if (position == std::string::npos || end == std::string::npos)
    throw std::runtime_error("invalid lock string " + key);
  return text.substr(position + 1, end - position - 1);
}

uint64_t ExtractUnsigned(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  size_t position = text.find(needle);
  if (position == std::string::npos) throw std::runtime_error("lock missing " + key);
  position = text.find(':', position + needle.size()) + 1;
  while (position < text.size() && std::isspace(text[position])) ++position;
  return std::stoull(text.substr(position));
}

double ExtractRatio(const std::string& text, const std::string& key,
                    bool* is_null) {
  const std::string needle = "\"" + key + "\"";
  size_t position = text.find(needle);
  if (position == std::string::npos) throw std::runtime_error("lock missing " + key);
  position = text.find(':', position + needle.size()) + 1;
  while (position < text.size() && std::isspace(text[position])) ++position;
  *is_null = text.compare(position, 4, "null") == 0;
  return *is_null ? 0.0 : std::stod(text.substr(position));
}

void WritePreflight(const std::string& path, uint64_t seed,
                    const cudaDeviceProp& property, int active_blocks_per_sm,
                    const std::vector<RoutePlan>& plans) {
  const int resident_capacity =
      property.multiProcessorCount * active_blocks_per_sm;
  double largest_medium = 0.0;
  double smallest_high = std::numeric_limits<double>::infinity();
  double largest_high = 0.0;
  for (const RoutePlan& plan : plans) {
    const double ratio = static_cast<double>(plan.tasks.size()) /
                         static_cast<double>(resident_capacity);
    if (plan.scale == "MEDIUM_RESIDENCY") largest_medium = std::max(largest_medium, ratio);
    else {
      smallest_high = std::min(smallest_high, ratio);
      largest_high = std::max(largest_high, ratio);
    }
  }
  const bool pass = largest_medium <= 0.65 && smallest_high >= 0.90 &&
                    largest_high <= 1.10;
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create preflight JSON");
  out << "{\n  \"schema\":\"" << kPreflightSchema << "\",\n"
      << "  \"status\":\""
      << (pass ? "PREFLIGHT_PASS_LOCKED" : "PREFLIGHT_NEEDS_SHAPE_ADJUSTMENT")
      << "\",\n  \"route_seed\":" << seed << ",\n"
      << "  \"sm_count\":" << property.multiProcessorCount << ",\n"
      << "  \"active_blocks_per_sm\":" << active_blocks_per_sm << ",\n"
      << "  \"resident_block_capacity\":" << resident_capacity << ",\n"
      << "  \"producer_block_size\":" << kProducerThreads << ",\n"
      << "  \"medium_tokens\":" << kMediumTokens << ",\n"
      << "  \"high_tokens\":" << kHighTokens << ",\n"
      << "  \"acceptance\":{\"medium_max_capacity_ratio\":0.65,"
         "\"high_min_capacity_ratio\":0.90,"
         "\"high_max_capacity_ratio\":1.10},\n"
      << "  \"cells\":[\n";
  for (size_t index = 0; index < plans.size(); ++index) {
    const RoutePlan& plan = plans[index];
    out << "    {\"cell_id\":\"" << plan.cell_id
        << "\",\"route_distribution\":\"" << plan.distribution
        << "\",\"problem_scale\":\"" << plan.scale
        << "\",\"producer_grid_size\":" << plan.tasks.size()
        << ",\"resident_capacity_ratio\":" << std::setprecision(10)
        << static_cast<double>(plan.tasks.size()) / resident_capacity
        << ",\"expert_routed_token_counts\":"
        << CountsJson(plan.expert_routed_token_counts)
        << ",\"task_order_contract\":\"expert_major_critical_first_within_expert_v1\"}"
        << (index + 1 == plans.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"comparison_executed\":false\n}\n";
}

void ValidatePreflightLock(const std::string& path,
                           const cudaDeviceProp& property,
                           int active_blocks_per_sm) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read preflight lock: " + path);
  std::ostringstream bytes;
  bytes << input.rdbuf();
  const std::string text = bytes.str();
  if (ExtractString(text, "schema") != kPreflightSchema ||
      ExtractString(text, "status") != "PREFLIGHT_PASS_LOCKED") {
    throw std::runtime_error("preflight lock is not passing/frozen");
  }
  if (ExtractUnsigned(text, "sm_count") !=
          static_cast<uint64_t>(property.multiProcessorCount) ||
      ExtractUnsigned(text, "active_blocks_per_sm") !=
          static_cast<uint64_t>(active_blocks_per_sm) ||
      ExtractUnsigned(text, "producer_block_size") != kProducerThreads ||
      ExtractUnsigned(text, "medium_tokens") != kMediumTokens ||
      ExtractUnsigned(text, "high_tokens") != kHighTokens) {
    throw std::runtime_error("current GPU/source shape differs from preflight lock");
  }
}

std::vector<int32_t> ExtractCounts(const std::string& text) {
  const std::string key = "\"expert_routed_token_counts\"";
  size_t begin = text.find('[', text.find(key));
  size_t end = text.find(']', begin);
  if (begin == std::string::npos || end == std::string::npos)
    throw std::runtime_error("lock missing expert counts");
  std::vector<int32_t> values;
  std::stringstream input(text.substr(begin + 1, end - begin - 1));
  std::string token;
  while (std::getline(input, token, ',')) values.push_back(std::stoi(token));
  return values;
}

RunLock ReadRunLock(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read run lock: " + path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  if (ExtractString(text, "schema") != kRunLockSchema)
    throw std::runtime_error("unexpected run lock schema");
  if (ExtractString(text, "status") != "LOCKED_BEFORE_FORMAL_RUN")
    throw std::runtime_error("run lock is not frozen before formal run");
  RunLock lock;
  lock.calibration_seed = ExtractUnsigned(text, "calibration_route_seed");
  lock.formal_seed = ExtractUnsigned(text, "formal_route_seed");
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("\"cell_id\"") == std::string::npos) continue;
    LockCell cell;
    cell.cell_id = ExtractString(line, "cell_id");
    cell.distribution = ExtractString(line, "route_distribution");
    cell.scale = ExtractString(line, "problem_scale");
    cell.status = ExtractString(line, "gate_selection_status");
    bool is_null = false;
    cell.gate_ratio = ExtractRatio(line, "gate_remaining_ratio", &is_null);
    if (is_null && cell.status != "NO_SAFE_GATE")
      throw std::runtime_error("selected gate cannot be null");
    cell.total_tokens = static_cast<int>(ExtractUnsigned(line, "total_tokens"));
    cell.routed_tokens = static_cast<int>(ExtractUnsigned(line, "routed_tokens"));
    cell.theoretical_flops = ExtractUnsigned(line, "theoretical_flops");
    cell.expert_routed_token_counts = ExtractCounts(line);
    lock.cells.push_back(std::move(cell));
  }
  if (lock.cells.size() != 4) throw std::runtime_error("run lock must contain 4 cells");
  return lock;
}

struct CandidateResult {
  double ratio = 0.0;
  double producer_regression_ratio = 0.0;
};

struct CalibrationCellResult {
  const RoutePlan* plan = nullptr;
  std::string status = "NO_SAFE_GATE";
  double selected_ratio = 0.0;
  std::vector<CandidateResult> candidates;
};

void WriteCalibrationJson(const std::string& path, uint64_t calibration_seed,
                          const std::vector<CalibrationCellResult>& cells) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create calibration JSON");
  out << "{\n  \"schema\": \"" << kCalibrationSchema << "\",\n"
      << "  \"status\": \"CALIBRATION_COMPLETE_LOCKED\",\n"
      << "  \"calibration_route_seed\": " << calibration_seed << ",\n"
      << "  \"gate_candidates_remaining_ratio\": [0,0.125,0.25,0.5],\n"
      << "  \"repetitions\": {\"warmups_per_cell_candidate\": "
      << kCalibrationWarmups << ", \"measured_per_cell_candidate\": "
      << kCalibrationRepeats << "},\n  \"cells\": [\n";
  for (size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    const auto& cell = cells[cell_index];
    out << "    {\"cell_id\":\"" << cell.plan->cell_id
        << "\",\"route_distribution\":\"" << cell.plan->distribution
        << "\",\"problem_scale\":\"" << cell.plan->scale
        << "\",\"selection_status\":\"" << cell.status << "\","
        << "\"selected_gate_remaining_ratio\":";
    if (cell.status == "SELECTED") out << cell.selected_ratio;
    else out << "null";
    out << ",\"candidates\":[";
    for (size_t candidate = 0; candidate < cell.candidates.size(); ++candidate) {
      if (candidate) out << ',';
      out << "{\"remaining_ratio\":" << cell.candidates[candidate].ratio
          << ",\"producer_regression_ratio\":"
          << cell.candidates[candidate].producer_regression_ratio << '}';
    }
    out << "]}" << (cell_index + 1 == cells.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

void WriteRunLock(const std::string& path, uint64_t calibration_seed,
                  uint64_t formal_seed,
                  const std::vector<CalibrationCellResult>& cells,
                  const std::string& calibration_file,
                  const std::string& preflight_file,
                  const cudaDeviceProp& property,
                  int active_blocks_per_sm) {
  std::ifstream calibration_input(calibration_file, std::ios::binary);
  std::ostringstream calibration_bytes;
  calibration_bytes << calibration_input.rdbuf();
  const std::string calibration_hash = Sha256Hex(calibration_bytes.str());
  std::ifstream preflight_input(preflight_file, std::ios::binary);
  std::ostringstream preflight_bytes;
  preflight_bytes << preflight_input.rdbuf();
  const std::string preflight_hash = Sha256Hex(preflight_bytes.str());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create run lock");
  out << "{\n  \"schema\": \"" << kRunLockSchema << "\",\n"
      << "  \"status\": \"LOCKED_BEFORE_FORMAL_RUN\",\n"
      << "  \"calibration_sha256\": \"" << calibration_hash << "\",\n"
      << "  \"preflight_sha256\": \"" << preflight_hash << "\",\n"
      << "  \"calibration_route_seed\": " << calibration_seed << ",\n"
      << "  \"formal_route_seed\": " << formal_seed << ",\n"
      << "  \"residency_lock\": {\"sm_count\":"
      << property.multiProcessorCount << ",\"active_blocks_per_sm\":"
      << active_blocks_per_sm << ",\"resident_block_capacity\":"
      << property.multiProcessorCount * active_blocks_per_sm
      << ",\"producer_block_size\":" << kProducerThreads << "},\n"
      << "  \"repetitions\": {\"warmups_per_cell_mode\": "
      << kFormalWarmups << ", \"correctness\": " << kCorrectnessRepeats
      << ", \"utility\": " << kUtilityRepeats << "},\n"
      << "  \"cells\": [\n";
  for (size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    out << "    {\"cell_id\":\"" << cell.plan->cell_id
        << "\",\"route_distribution\":\"" << cell.plan->distribution
        << "\",\"problem_scale\":\"" << cell.plan->scale
        << "\",\"gate_selection_status\":\"" << cell.status
        << "\",\"gate_remaining_ratio\":";
    if (cell.status == "SELECTED") out << cell.selected_ratio;
    else out << "null";
    out << ",\"total_tokens\":" << cell.plan->total_tokens
        << ",\"routed_tokens\":" << cell.plan->routed_token_ids.size()
        << ",\"top_k\":" << kTopK << ",\"expert_count\":" << kExperts
        << ",\"theoretical_flops\":" << cell.plan->theoretical_flops
        << ",\"producer_grid_size\":" << cell.plan->tasks.size()
        << ",\"task_order_contract\":\"expert_major_critical_first_within_expert_v1\""
        << ",\"expert_routed_token_counts\":"
        << CountsJson(cell.plan->expert_routed_token_counts) << "}"
        << (index + 1 == cells.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"contracts\": {"
         "\"synthetic_delay\":false,"
         "\"artificial_sm_reservation\":false,"
         "\"single_producer_kernel\":true,"
         "\"matrix_multiply_like_work\":true,"
         "\"representative_grouped_expert\":true}\n}\n";
}

void WriteMeta(const std::string& path, const Options& options,
               uint64_t input_hash, uint64_t rows) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot create meta JSON");
  out << "{\n  \"schema\": \"" << kMetaSchema << "\",\n"
      << "  \"status\": \"RAW_COMPLETE_NOT_ADJUDICATED\",\n"
      << "  \"phase\": \""
      << (options.phase == Phase::kCalibration ? "calibration" : "formal")
      << "\",\n  \"route_seed\": " << options.seed << ",\n"
      << "  \"input_hash\": " << input_hash << ",\n"
      << "  \"raw_rows\": " << rows << ",\n"
      << "  \"workload\": {\"dtype\":\"FP16\",\"hidden\":"
      << kHidden << ",\"output\":" << kOutput
      << ",\"top_k\":2,\"experts\":8,\"rows_per_tile\":64},\n"
      << "  \"contracts\": {\"synthetic_delay\":false,"
         "\"artificial_sm_reservation\":false,"
         "\"producer_launches_per_trial\":1,"
         "\"consumer_launches_per_trial\":1}\n}\n";
}

double ProducerDuration(const TrialRecord& row) {
  return static_cast<double>(row.producer_end_ns - row.producer_start_ns);
}

std::vector<CalibrationCellResult> RunCalibration(
    const Options& options, const DeviceBuffers& buffers, const Streams& streams,
    const std::vector<RoutePlan>& plans, uint64_t input_hash,
    std::ostream& raw) {
  std::vector<CalibrationCellResult> results;
  const std::array<double, 4> earliest_order{{0.5, 0.25, 0.125, 0.0}};
  for (const RoutePlan& plan : plans) {
    UploadPlan(buffers, plan);
    CalibrationCellResult cell;
    cell.plan = &plan;
    for (double ratio : kGateCandidates) {
      std::vector<double> regressions;
      for (int phase_index = 0; phase_index < 2; ++phase_index) {
        const bool warmup = phase_index == 0;
        const int repeats = warmup ? kCalibrationWarmups : kCalibrationRepeats;
        for (int repeat = 0; repeat < repeats; ++repeat) {
          const bool ac_order = repeat % 2 == 0;
          const std::string order = ac_order ? "AC" : "CA";
          TrialRecord baseline;
          TrialRecord gated;
          for (int slot = 0; slot < 2; ++slot) {
            const Variant variant =
                (slot == 0) == ac_order ? Variant::kAllDoneSham
                                        : Variant::kProgressGated;
            TrialRecord row = RunTrial(
                buffers, streams, plan, input_hash, options.seed,
                Phase::kCalibration, WorkMode::kUtility,
                warmup ? "warmup" : "measured", repeat, slot, order, variant,
                ratio, "CALIBRATION_CANDIDATE");
            if (variant == Variant::kAllDoneSham) baseline = std::move(row);
            else gated = std::move(row);
          }
          FinalizePairOrTriple({&baseline, &gated});
          WriteCsvRow(raw, baseline);
          WriteCsvRow(raw, gated);
          if (!warmup) {
            regressions.push_back((ProducerDuration(gated) -
                                   ProducerDuration(baseline)) /
                                  ProducerDuration(baseline));
          }
        }
      }
      cell.candidates.push_back(CandidateResult{ratio, Median(regressions)});
    }
    for (double ratio : earliest_order) {
      const auto found = std::find_if(
          cell.candidates.begin(), cell.candidates.end(),
          [&](const CandidateResult& candidate) { return candidate.ratio == ratio; });
      if (found != cell.candidates.end() &&
          found->producer_regression_ratio < 0.05) {
        cell.status = "SELECTED";
        cell.selected_ratio = ratio;
        break;
      }
    }
    results.push_back(std::move(cell));
    raw.flush();
    std::cout << "calibrated " << plan.cell_id << '\n';
  }
  return results;
}

const std::array<std::array<Variant, 3>, 6>& Permutations() {
  static const std::array<std::array<Variant, 3>, 6> values{{
      {{Variant::kAllDoneSham, Variant::kEager, Variant::kProgressGated}},
      {{Variant::kAllDoneSham, Variant::kProgressGated, Variant::kEager}},
      {{Variant::kEager, Variant::kAllDoneSham, Variant::kProgressGated}},
      {{Variant::kEager, Variant::kProgressGated, Variant::kAllDoneSham}},
      {{Variant::kProgressGated, Variant::kAllDoneSham, Variant::kEager}},
      {{Variant::kProgressGated, Variant::kEager, Variant::kAllDoneSham}},
  }};
  return values;
}

std::string PermutationName(const std::array<Variant, 3>& values) {
  std::string result;
  for (Variant variant : values)
    result.push_back(variant == Variant::kAllDoneSham
                         ? 'A'
                         : (variant == Variant::kEager ? 'B' : 'C'));
  return result;
}

uint64_t RunFormal(const Options& options, const DeviceBuffers& buffers,
                   const Streams& streams, const std::vector<RoutePlan>& plans,
                   const RunLock& lock, uint64_t input_hash,
                   std::ostream& raw) {
  if (lock.formal_seed != options.seed ||
      lock.calibration_seed == options.seed) {
    throw std::runtime_error("formal seed does not match independent run lock");
  }
  uint64_t rows_written = 0;
  const auto& permutations = Permutations();
  for (const RoutePlan& plan : plans) {
    const auto lock_cell = std::find_if(
        lock.cells.begin(), lock.cells.end(),
        [&](const LockCell& cell) { return cell.cell_id == plan.cell_id; });
    if (lock_cell == lock.cells.end()) throw std::runtime_error("lock cell missing");
    if (lock_cell->total_tokens != plan.total_tokens ||
        lock_cell->routed_tokens != static_cast<int>(plan.routed_token_ids.size()) ||
        lock_cell->theoretical_flops != plan.theoretical_flops ||
        lock_cell->expert_routed_token_counts !=
            plan.expert_routed_token_counts) {
      throw std::runtime_error("formal route/workload differs from run lock");
    }
    const double gate_ratio =
        lock_cell->status == "NO_SAFE_GATE" ? 0.0 : lock_cell->gate_ratio;
    UploadPlan(buffers, plan);
    for (WorkMode mode : {WorkMode::kCorrectness, WorkMode::kUtility}) {
      for (int phase_index = 0; phase_index < 2; ++phase_index) {
        const bool warmup = phase_index == 0;
        const int repeats = warmup
                                ? kFormalWarmups
                                : (mode == WorkMode::kCorrectness
                                       ? kCorrectnessRepeats
                                       : kUtilityRepeats);
        for (int repeat = 0; repeat < repeats; ++repeat) {
          const auto& permutation = permutations[repeat % permutations.size()];
          const std::string order = PermutationName(permutation);
          std::array<TrialRecord, 3> trial_rows;
          for (int slot = 0; slot < 3; ++slot) {
            const Variant variant = permutation[slot];
            trial_rows[static_cast<int>(variant)] = RunTrial(
                buffers, streams, plan, input_hash, options.seed,
                Phase::kFormal, mode, warmup ? "warmup" : "measured", repeat,
                slot, order, variant, gate_ratio, lock_cell->status);
          }
          FinalizePairOrTriple(
              {&trial_rows[0], &trial_rows[1], &trial_rows[2]});
          for (TrialRecord& row : trial_rows) {
            WriteCsvRow(raw, row);
            ++rows_written;
          }
        }
      }
    }
    raw.flush();
    std::cout << "formal completed " << plan.cell_id << '\n';
  }
  return rows_written;
}

}  // namespace real_moe_tail

int main(int argc, char** argv) {
  using namespace real_moe_tail;
  try {
    const Options options = ParseOptions(argc, argv);
    if (Sha256Hex("abc") !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
      throw std::runtime_error("internal SHA-256 self-test failed");
    }

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || options.device < 0 ||
        options.device >= device_count) {
      const std::string reason =
          count_status == cudaSuccess ? "requested CUDA device is unavailable"
                                      : cudaGetErrorString(count_status);
      WriteBlocked(options.output_prefix, reason);
      std::cerr << "BLOCKED_NO_GPU: " << reason << '\n';
      return 2;
    }
    CUDA_THROW(cudaSetDevice(options.device));
    cudaDeviceProp property{};
    CUDA_THROW(cudaGetDeviceProperties(&property, options.device));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_THROW(cudaDriverGetVersion(&driver_version));
    CUDA_THROW(cudaRuntimeGetVersion(&runtime_version));

    int active_blocks_per_sm = 0;
    CUDA_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, GroupedExpertProducer, kProducerThreads, 0));
    if (active_blocks_per_sm <= 0)
      throw std::runtime_error("grouped producer occupancy is zero");

    const std::vector<RoutePlan> plans = BuildFourPlans(options.seed);
    if (options.phase == Phase::kPreflight) {
      uint64_t* timer_probe = nullptr;
      CUDA_THROW(cudaMalloc(&timer_probe, sizeof(uint64_t)));
      const uint64_t timer_resolution = MeasureTimerResolution(timer_probe);
      CUDA_THROW(cudaFree(timer_probe));
      WriteEnvironment(options.output_prefix + ".environment.json", property,
                       driver_version, runtime_version, timer_resolution);
      const std::string preflight_path =
          options.output_prefix + ".preflight.json";
      WritePreflight(preflight_path, options.seed, property,
                     active_blocks_per_sm, plans);
      const int resident_capacity =
          property.multiProcessorCount * active_blocks_per_sm;
      double largest_medium = 0.0;
      double smallest_high = std::numeric_limits<double>::infinity();
      double largest_high = 0.0;
      for (const RoutePlan& plan : plans) {
        const double ratio = static_cast<double>(plan.tasks.size()) /
                             resident_capacity;
        if (plan.scale == "MEDIUM_RESIDENCY")
          largest_medium = std::max(largest_medium, ratio);
        else {
          smallest_high = std::min(smallest_high, ratio);
          largest_high = std::max(largest_high, ratio);
        }
      }
      const bool pass = largest_medium <= 0.65 && smallest_high >= 0.90 &&
                        largest_high <= 1.10;
      std::cout << (pass ? "PREFLIGHT_PASS_LOCKED: "
                         : "PREFLIGHT_NEEDS_SHAPE_ADJUSTMENT: ")
                << preflight_path << '\n';
      return pass ? 0 : 4;
    }

    if (options.phase == Phase::kCalibration) {
      ValidatePreflightLock(options.preflight_lock_path, property,
                            active_blocks_per_sm);
    }

    const int maximum_tasks = static_cast<int>(
        std::max_element(plans.begin(), plans.end(),
                         [](const RoutePlan& lhs, const RoutePlan& rhs) {
                           return lhs.tasks.size() < rhs.tasks.size();
                         })->tasks.size());
    DeviceBuffers buffers;
    Streams streams;
    AllocateBuffers(&buffers, kHighTokens, kHighTokens * kTopK,
                    maximum_tasks);
    CreateStreams(&streams);
    constexpr uint64_t kFixedInputSeed = 0x494e5055545f5631ull;
    const uint64_t input_hash =
        UploadFixedInputs(&buffers, kHighTokens, kFixedInputSeed);
    const uint64_t timer_resolution =
        MeasureTimerResolution(buffers.timer_probe);
    WriteEnvironment(options.output_prefix + ".environment.json", property,
                     driver_version, runtime_version, timer_resolution);

    const std::string raw_path = options.output_prefix + ".raw.csv";
    std::ofstream raw(raw_path);
    if (!raw) throw std::runtime_error("cannot create raw CSV");
    WriteCsvHeader(raw);
    uint64_t rows_written = 0;
    if (options.phase == Phase::kCalibration) {
      const std::vector<CalibrationCellResult> calibration = RunCalibration(
          options, buffers, streams, plans, input_hash, raw);
      rows_written = static_cast<uint64_t>(plans.size()) *
                     kGateCandidates.size() *
                     (kCalibrationWarmups + kCalibrationRepeats) * 2ull;
      raw.flush();
      const std::string calibration_path =
          options.output_prefix + ".calibration.json";
      WriteCalibrationJson(calibration_path, options.seed, calibration);
      WriteRunLock(options.output_prefix + ".run_lock.json", options.seed,
                   options.formal_seed, calibration, calibration_path,
                   options.preflight_lock_path, property,
                   active_blocks_per_sm);
    } else {
      const RunLock lock = ReadRunLock(options.run_lock_path);
      rows_written = RunFormal(options, buffers, streams, plans, lock,
                               input_hash, raw);
    }
    raw.close();
    WriteMeta(options.output_prefix + ".meta.json", options, input_hash,
              rows_written);
    DestroyStreams(&streams);
    FreeBuffers(&buffers);
    std::cout << "RAW_COMPLETE_NOT_ADJUDICATED: " << raw_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 3;
  }
}
