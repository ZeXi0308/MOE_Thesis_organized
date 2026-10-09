// Asynchronous PER-REQUEST output encoding. Fixed pools, no submit allocation,
// no host event/stream/device synchronization, and no cross-request prefix scan.
// Initialization/destruction allocate/free CUDA resources outside the hot path.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int COLS = 21, TOP_K = 20, NUM_BYTES = 192, MAX_LIMBS = 20;
constexpr uint64_t BASE = 1000000000ULL;
constexpr int INTRO_LEN = sizeof(",\"top_logprobs\":[") - 1;
thread_local std::string last_error;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

struct RowHeader {
  uint32_t status, json_length, columns, reserved;
  int64_t rank;
  int64_t ids[COLS];
  float values[COLS];
};
static_assert(sizeof(RowHeader) == 280, "Python RowHeader ABI must match");

struct Metrics {
  uint64_t rows, snapshot_bytes, transferred_bytes, header_bytes, valid_json_bytes;
  float snapshot_ms, kernel_ms, d2h_ms, sampling_ready_to_host_ms;
  uint32_t encoded, id_bytes, rank_bytes, reserved;
};

struct Slot {
  bool busy = false;
  uint32_t generation = 0;
  int rows = 0, id_bytes = 0, rank_bytes = 0;
  bool encode = true;
  uint64_t snapshot_bytes = 0;
  cudaEvent_t sampling_ready = nullptr, snapshot_ready = nullptr;
  cudaEvent_t kernel_start = nullptr, kernel_end = nullptr, done = nullptr;
};

struct Context {
  int device = 0, capacity = 0, max_rows = 0, stream_count = 0, next_slot = 0;
  bool poisoned = false;
  uint64_t vocab = 0, stride = 0, json_capacity = 0, device_bytes = 0, pinned_bytes = 0;
  char *table = nullptr, *ids = nullptr, *ranks = nullptr, *output = nullptr, *host = nullptr;
  float *values = nullptr;
  uint64_t *table_offsets = nullptr;
  uint32_t *table_lengths = nullptr;
  uint8_t *unsafe = nullptr;
  std::vector<Slot> slots;
  std::vector<cudaStream_t> streams;
  ~Context() {
    cudaSetDevice(device);
    for (auto &s : slots) {
      if (s.sampling_ready) cudaEventDestroy(s.sampling_ready);
      if (s.snapshot_ready) cudaEventDestroy(s.snapshot_ready);
      if (s.kernel_start) cudaEventDestroy(s.kernel_start);
      if (s.kernel_end) cudaEventDestroy(s.kernel_end);
      if (s.done) cudaEventDestroy(s.done);
    }
    for (auto s : streams) if (s) cudaStreamDestroy(s);
    cudaFree(table); cudaFree(table_offsets); cudaFree(table_lengths); cudaFree(unsafe);
    cudaFree(ids); cudaFree(values); cudaFree(ranks); cudaFree(output); cudaFreeHost(host);
  }
};

template <class T> void alloc_device(Context *c, T **p, uint64_t count) {
  check(cudaMalloc(reinterpret_cast<void **>(p), count * sizeof(T)));
  c->device_bytes += count * sizeof(T);
}

// Identical exact-float arithmetic to gpu_encoder.cu; no cached probabilities,
// decimal quantization or fast-math. JSON spelling can exceed CPU shortest form.
__device__ int decimal_u32(uint32_t n, char *dst, int pad = 0) {
  char reversed[10]; int count = 0;
  do { reversed[count++] = char('0' + n % 10); n /= 10; } while (n);
  while (count < pad) reversed[count++] = '0';
  for (int j = 0; j < count; ++j) dst[j] = reversed[count - j - 1];
  return count;
}
__device__ int exact_float(float input, char *dst) {
  if (input < -9999.0f) input = -9999.0f;
  uint32_t bits = __float_as_uint(input), exponent = (bits >> 23) & 255U;
  if (exponent == 255U) return -1;
  uint32_t mantissa = bits & 0x7fffffU;
  int pos = 0;
  if (bits >> 31) dst[pos++] = '-';
  if (exponent == 0 && mantissa == 0) {
    dst[pos++] = '0'; dst[pos++] = '.'; dst[pos++] = '0'; return pos;
  }
  int exp2 = exponent ? int(exponent) - 127 - 23 : -149;
  if (exponent) mantissa |= 0x800000U;
  while (exp2 < 0 && (mantissa & 1U) == 0) { mantissa >>= 1; ++exp2; }
  uint32_t limbs[MAX_LIMBS]; limbs[0] = mantissa; int nlimbs = 1;
  int steps = exp2 < 0 ? -exp2 : exp2;
  uint32_t multiplier = exp2 < 0 ? 5U : 2U;
  for (int step = 0; step < steps; ++step) {
    uint64_t carry = 0;
    for (int j = 0; j < nlimbs; ++j) {
      uint64_t value = uint64_t(limbs[j]) * multiplier + carry;
      limbs[j] = uint32_t(value % BASE); carry = value / BASE;
    }
    if (carry) limbs[nlimbs++] = uint32_t(carry);
  }
  char digits[180]; int ndigits = decimal_u32(limbs[nlimbs - 1], digits);
  for (int j = nlimbs - 2; j >= 0; --j) ndigits += decimal_u32(limbs[j], digits + ndigits, 9);
  if (exp2 >= 0) {
    for (int j = 0; j < ndigits; ++j) dst[pos++] = digits[j];
    dst[pos++] = '.'; dst[pos++] = '0';
  } else {
    int integer_digits = ndigits + exp2;
    if (integer_digits <= 0) {
      dst[pos++] = '0'; dst[pos++] = '.';
      for (int j = 0; j < -integer_digits; ++j) dst[pos++] = '0';
      for (int j = 0; j < ndigits; ++j) dst[pos++] = digits[j];
    } else {
      for (int j = 0; j < ndigits; ++j) {
        if (j == integer_digits) dst[pos++] = '.';
        dst[pos++] = digits[j];
      }
    }
  }
  return pos;
}

__device__ uint32_t copy_bytes(char *out, uint32_t pos, const char *src, uint32_t count) {
  for (uint32_t j = 0; j < count; ++j) out[pos + j] = src[j];
  return pos + count;
}

// A request owns this launch. Multiple ready rows from this SAME request use
// independent fixed row slots. Only that request's event waits for this launch.
template <class ID, class RANK>
__global__ void encode_request(const ID *ids, const float *values, const RANK *ranks,
    const char *table, const uint64_t *table_offsets, const uint32_t *table_lengths,
    const uint8_t *unsafe, uint64_t vocab, char *output, uint64_t stride,
    uint32_t json_capacity) {
  int row = int(blockIdx.x), lane = int(threadIdx.x);
  char *slot = output + uint64_t(row) * stride;
  // Padding really crosses PCIe. Initialize it rather than expose stale bytes
  // from prior requests. These stores are intentionally part of kernel cost.
  for (uint64_t j = lane; j < stride; j += blockDim.x) slot[j] = 0;
  __syncthreads();
  RowHeader *header = reinterpret_cast<RowHeader *>(slot);
  char *out = slot + sizeof(RowHeader);
  __shared__ char numbers[COLS][NUM_BYTES];
  __shared__ uint32_t lengths[COLS], starts[COLS];
  __shared__ int selected[COLS], top_count;
  __shared__ uint8_t entry_status[COLS];
  if (lane < COLS) {
    int64_t id = int64_t(ids[row * COLS + lane]);
    float value = values[row * COLS + lane];
    header->ids[lane] = id; header->values[lane] = value;
    uint8_t state = 0; int length = 0;
    if (id < 0 || uint64_t(id) >= vocab) state = 2;
    else if (unsafe[id]) state = 1;
    else { length = exact_float(value, numbers[lane]); if (length < 0) { state = 3; length = 0; } }
    entry_status[lane] = state; lengths[lane] = uint32_t(length);
  }
  if (lane == 0) { header->rank = int64_t(ranks[row]); header->columns = COLS; }
  __syncthreads();
  if (lane == 0) {
    for (int j = 0; j < COLS; ++j)
      if (entry_status[j]) { header->status = entry_status[j]; break; }
    if (!header->status) {
      int unique[COLS], count = 0;
      for (int j = 0; j < COLS; ++j) {
        bool first = true;
        for (int k = 0; k < j; ++k) if (header->ids[k] == header->ids[j]) { first = false; break; }
        if (!first) continue;
        int last = j;
        for (int k = j + 1; k < COLS; ++k) if (header->ids[k] == header->ids[j]) last = k;
        unique[count++] = last;
      }
      top_count = count < TOP_K ? count : TOP_K;
      selected[0] = unique[0]; starts[0] = 0;
      uint64_t id = uint64_t(header->ids[unique[0]]);
      uint32_t at = table_lengths[id * 2] + table_lengths[id * 2 + 1] + lengths[unique[0]] + INTRO_LEN;
      for (int j = 0; j < top_count; ++j) {
        int source = unique[j]; id = uint64_t(header->ids[source]);
        selected[j + 1] = source; starts[j + 1] = at;
        at += table_lengths[id * 2] + table_lengths[id * 2 + 1] + lengths[source] + 1;
        if (j + 1 < top_count) ++at;
      }
      if (uint64_t(at) + 2 > json_capacity) header->status = 4;
      else header->json_length = at + 2;
    }
  }
  __syncthreads();
  if (header->status) return;
  if (lane <= top_count) {
    int source = selected[lane]; uint64_t id = uint64_t(header->ids[source]);
    uint32_t at = starts[lane];
    at = copy_bytes(out, at, table + table_offsets[id * 2], table_lengths[id * 2]);
    at = copy_bytes(out, at, numbers[source], lengths[source]);
    at = copy_bytes(out, at, table + table_offsets[id * 2 + 1], table_lengths[id * 2 + 1]);
    if (lane == 0) { const char intro[] = ",\"top_logprobs\":["; copy_bytes(out, at, intro, INTRO_LEN); }
    else { out[at++] = '}'; if (lane < top_count) out[at] = ','; }
  }
  if (lane == 0) { out[header->json_length - 2] = ']'; out[header->json_length - 1] = '}'; }
}

Slot &slot_for(Context *c, uint64_t ticket, int *index = nullptr) {
  uint32_t at = uint32_t(ticket), generation = uint32_t(ticket >> 32);
  if (at >= uint32_t(c->capacity) || !c->slots[at].busy || c->slots[at].generation != generation)
    throw std::runtime_error("stale or released ticket");
  if (index) *index = int(at);
  return c->slots[at];
}

int ready(Slot &s) {
  auto e = cudaEventQuery(s.done);
  if (e == cudaErrorNotReady) return 0;
  check(e); return 1;
}
}  // namespace

extern "C" {
const char *f_async_error() { return last_error.c_str(); }
uint32_t f_async_abi() { return 1; }

void *f_async_create(const char *blob, uint64_t blob_bytes, const uint64_t *offsets,
    const uint32_t *lengths, const uint8_t *unsafe, uint64_t vocab,
    int capacity, int max_rows, int stream_count, int device) {
  Context *c = nullptr;
  try {
    if (!vocab || capacity < 1 || capacity > 4096 || max_rows < 1 || max_rows > 4096 ||
        stream_count < 1 || stream_count > capacity)
      throw std::runtime_error("invalid fixed pool configuration");
    check(cudaSetDevice(device));
    c = new Context(); c->device = device; c->capacity = capacity; c->max_rows = max_rows;
    c->stream_count = stream_count; c->vocab = vocab;
    uint64_t max_piece = 0;
    for (uint64_t i = 0; i < vocab; ++i) {
      for (int side = 0; side < 2; ++side)
        if (offsets[i*2+side] > blob_bytes || lengths[i*2+side] > blob_bytes-offsets[i*2+side])
          throw std::runtime_error("token table span out of bounds");
      max_piece = std::max(max_piece, uint64_t(lengths[i*2]) + lengths[i*2+1]);
    }
    c->json_capacity = (max_piece + NUM_BYTES + 2) * COLS + INTRO_LEN + 2;
    c->stride = (sizeof(RowHeader) + c->json_capacity + 63) & ~uint64_t(63);
    uint64_t all_rows = uint64_t(capacity) * max_rows;
    if (c->json_capacity > UINT32_MAX || all_rows*c->stride > (uint64_t(1)<<34))
      throw std::runtime_error("fixed slots exceed supported memory bound");
    alloc_device(c, &c->table, std::max(blob_bytes, uint64_t(1)));
    alloc_device(c, &c->table_offsets, vocab*2); alloc_device(c, &c->table_lengths, vocab*2);
    alloc_device(c, &c->unsafe, vocab);
    check(cudaMemcpy(c->table, blob, blob_bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->table_offsets, offsets, vocab*2*sizeof(uint64_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->table_lengths, lengths, vocab*2*sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->unsafe, unsafe, vocab, cudaMemcpyHostToDevice));
    alloc_device(c, &c->ids, all_rows*COLS*8); alloc_device(c, &c->values, all_rows*COLS);
    alloc_device(c, &c->ranks, all_rows*8); alloc_device(c, &c->output, all_rows*c->stride);
    check(cudaMallocHost(reinterpret_cast<void **>(&c->host), all_rows*c->stride));
    c->pinned_bytes = all_rows*c->stride;
    c->streams.resize(stream_count, nullptr);
    for (auto &s : c->streams) check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    c->slots.resize(capacity);
    for (auto &s : c->slots) {
      check(cudaEventCreate(&s.sampling_ready)); check(cudaEventCreate(&s.snapshot_ready));
      check(cudaEventCreate(&s.kernel_start)); check(cudaEventCreate(&s.kernel_end));
      check(cudaEventCreate(&s.done));
    }
    return c;
  } catch (const std::exception &e) { last_error = e.what(); delete c; return nullptr; }
}

uint64_t f_async_device_bytes(void *p) { return static_cast<Context *>(p)->device_bytes; }
uint64_t f_async_pinned_bytes(void *p) { return static_cast<Context *>(p)->pinned_bytes; }
uint64_t f_async_row_stride(void *p) { return static_cast<Context *>(p)->stride; }
uint64_t f_async_header_bytes() { return sizeof(RowHeader); }

// 0 accepted; 1 pool full; 2 request exceeds fixed rows; 3 producer capture;
// -1 fatal error. No allocation or host synchronization in this function.
int f_async_submit(void *p, const void *ids, int id_bytes, const float *values,
    const void *ranks, int rank_bytes, int rows, int encode, uint64_t producer_ptr, uint64_t *ticket) {
  Context *c = static_cast<Context *>(p);
  bool reserved = false;
  try {
    if (!c || c->poisoned || !ids || !values || !ranks || rows < 1 ||
        (id_bytes != 4 && id_bytes != 8) || (rank_bytes != 4 && rank_bytes != 8))
      throw std::runtime_error("invalid input or poisoned async encoder");
    if (rows > c->max_rows) return 2;
    int index = -1;
    for (int j = 0; j < c->capacity; ++j) {
      int at = (c->next_slot+j) % c->capacity;
      if (!c->slots[at].busy) { index = at; break; }
    }
    if (index < 0) return 1;
    check(cudaSetDevice(c->device));
    cudaStream_t producer = reinterpret_cast<cudaStream_t>(producer_ptr);
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(producer, &capture));
    if (capture != cudaStreamCaptureStatusNone) return 3;
    Slot &s = c->slots[index];
    s.busy = true; reserved = true; s.rows = rows; s.id_bytes = id_bytes; s.rank_bytes = rank_bytes;
    s.encode = bool(encode);
    ++s.generation; if (!s.generation) ++s.generation;
    *ticket = (uint64_t(s.generation)<<32) | uint32_t(index);
    c->next_slot = (index+1) % c->capacity;
    uint64_t row_base = uint64_t(index)*c->max_rows;
    char *snapshot_ids = c->ids + row_base*COLS*8;
    float *snapshot_values = c->values + row_base*COLS;
    char *snapshot_ranks = c->ranks + row_base*8;
    char *out = c->output + row_base*c->stride;
    char *host = c->host + row_base*c->stride;
    s.snapshot_bytes = uint64_t(rows)*(COLS*(id_bytes+sizeof(float))+rank_bytes);
    // Snapshot is ordered ON THE PRODUCER STREAM, before the caller can enqueue
    // its next overwrite. Merely holding a tensor reference would not protect
    // reused graph/output storage. This D2D delays later producer work and is
    // an explicit measured cost, not "free overlap".
    check(cudaEventRecord(s.sampling_ready, producer));
    check(cudaMemcpyAsync(snapshot_ids, ids, uint64_t(rows)*COLS*id_bytes, cudaMemcpyDeviceToDevice, producer));
    check(cudaMemcpyAsync(snapshot_values, values, uint64_t(rows)*COLS*sizeof(float), cudaMemcpyDeviceToDevice, producer));
    check(cudaMemcpyAsync(snapshot_ranks, ranks, uint64_t(rows)*rank_bytes, cudaMemcpyDeviceToDevice, producer));
    check(cudaEventRecord(s.snapshot_ready, producer));
    cudaStream_t stream = c->streams[index % c->stream_count];
    check(cudaStreamWaitEvent(stream, s.snapshot_ready, 0));
    if(s.encode) {
      check(cudaEventRecord(s.kernel_start, stream));
#define LAUNCH(ID, RANK) encode_request<<<rows,32,0,stream>>>( \
    reinterpret_cast<const ID *>(snapshot_ids), snapshot_values, \
    reinterpret_cast<const RANK *>(snapshot_ranks), c->table, c->table_offsets, \
    c->table_lengths, c->unsafe, c->vocab, out, c->stride, uint32_t(c->json_capacity))
    if (id_bytes == 4 && rank_bytes == 4) { LAUNCH(int32_t, int32_t); }
    else if (id_bytes == 4) { LAUNCH(int32_t, int64_t); }
    else if (rank_bytes == 4) { LAUNCH(int64_t, int32_t); }
    else { LAUNCH(int64_t, int64_t); }
#undef LAUNCH
      check(cudaGetLastError());
      check(cudaEventRecord(s.kernel_end, stream));
      check(cudaMemcpyAsync(host, out, uint64_t(rows)*c->stride, cudaMemcpyDeviceToHost, stream));
    } else {
      // Strong CPU transport: exactly the original compact arrays, no encoding
      // kernel, normalization, per-row padding, or expanded JSON D2H.
      check(cudaEventRecord(s.kernel_end, stream));
      uint64_t ids_bytes=uint64_t(rows)*COLS*id_bytes;
      uint64_t values_bytes=uint64_t(rows)*COLS*sizeof(float);
      check(cudaMemcpyAsync(host,snapshot_ids,ids_bytes,cudaMemcpyDeviceToHost,stream));
      check(cudaMemcpyAsync(host+ids_bytes,snapshot_values,values_bytes,cudaMemcpyDeviceToHost,stream));
      check(cudaMemcpyAsync(host+ids_bytes+values_bytes,snapshot_ranks,
          uint64_t(rows)*rank_bytes,cudaMemcpyDeviceToHost,stream));
    }
    check(cudaEventRecord(s.done, stream));
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what(); if (c && reserved) c->poisoned = true; return -1;
  }
}

int f_async_poll(void *p, uint64_t ticket) {
  try { auto *c=static_cast<Context *>(p); check(cudaSetDevice(c->device)); return ready(slot_for(c,ticket)); }
  catch (const std::exception &e) { last_error=e.what(); return -1; }
}

// 1 ready, 0 pending, -1 error. The returned view is valid until release.
int f_async_collect_view(void *p, uint64_t ticket, const char **host, uint64_t *stride, Metrics *m) {
  try {
    auto *c=static_cast<Context *>(p); check(cudaSetDevice(c->device));
    int index; Slot &s=slot_for(c,ticket,&index); if (!ready(s)) return 0;
    *host=c->host+uint64_t(index)*c->max_rows*c->stride; *stride=c->stride;
    m->rows=s.rows; m->snapshot_bytes=s.snapshot_bytes;
    m->encoded=s.encode; m->id_bytes=s.id_bytes; m->rank_bytes=s.rank_bytes; m->reserved=0;
    m->transferred_bytes=s.encode ? uint64_t(s.rows)*c->stride : s.snapshot_bytes;
    m->header_bytes=s.encode ? uint64_t(s.rows)*sizeof(RowHeader) : 0;
    m->valid_json_bytes=0;
    for(int r=0;s.encode && r<s.rows;++r) {
      auto *h=reinterpret_cast<const RowHeader *>(*host+uint64_t(r)*c->stride);
      if(h->columns!=COLS || h->json_length>c->json_capacity)
        throw std::runtime_error("corrupt completed row metadata");
      m->valid_json_bytes+=h->json_length;
    }
    check(cudaEventElapsedTime(&m->snapshot_ms,s.sampling_ready,s.snapshot_ready));
    m->kernel_ms=0;
    if(s.encode) check(cudaEventElapsedTime(&m->kernel_ms,s.kernel_start,s.kernel_end));
    check(cudaEventElapsedTime(&m->d2h_ms,s.kernel_end,s.done));
    check(cudaEventElapsedTime(&m->sampling_ready_to_host_ms,s.sampling_ready,s.done));
    return 1;
  } catch (const std::exception &e) { last_error=e.what(); return -1; }
}

// 1 released, 0 still running. Cancellation must keep polling; no early reuse.
int f_async_release(void *p, uint64_t ticket) {
  try {
    auto *c=static_cast<Context *>(p); check(cudaSetDevice(c->device));
    Slot &s=slot_for(c,ticket); if(!ready(s)) return 0; s.busy=false; return 1;
  } catch(const std::exception &e) { last_error=e.what(); return -1; }
}

// Destruction is refused while any ticket is outstanding, even if complete.
// The owner must collect/drop and release each ticket explicitly.
int f_async_destroy(void *p) {
  auto *c=static_cast<Context *>(p);
  if(!c) return 1;
  for(const auto &s:c->slots) if(s.busy) return 0;
  delete c; return 1;
}
}  // extern C
