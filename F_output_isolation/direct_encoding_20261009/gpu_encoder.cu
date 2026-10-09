// Direct encoding of one already-ready sampling batch. No model or sampler changes.
// Build explicitly with gpu_encoder.py --build; importing the Python module is CPU-only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
constexpr int MAX_COLS = 33;
constexpr int NUM_BYTES = 192;  // exact decimal expansion of any finite float32
constexpr int MAX_LIMBS = 20;
constexpr uint64_t BASE = 1000000000ULL;
constexpr char TOP_INTRO[] = ",\"top_logprobs\":[";
constexpr int TOP_INTRO_LEN = sizeof(TOP_INTRO) - 1;
thread_local std::string last_error;

void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

struct Context {
  int device = 0, max_rows = 0;
  uint64_t vocab = 0, output_capacity = 0, device_bytes = 0, pinned_bytes = 0;
  char *table = nullptr, *numbers = nullptr, *output = nullptr, *host_output = nullptr;
  uint64_t *table_offsets = nullptr, *row_offsets = nullptr, *host_offsets = nullptr;
  uint32_t *table_lengths = nullptr, *number_lengths = nullptr, *row_lengths = nullptr;
  uint32_t *fragment_starts = nullptr;
  int32_t *selected = nullptr, *top_counts = nullptr;
  uint8_t *unsafe = nullptr, *status = nullptr, *host_status = nullptr;
  cudaEvent_t kernel_start = nullptr, kernel_end = nullptr;
  cudaEvent_t meta_start = nullptr, meta_end = nullptr;
  cudaEvent_t body_start = nullptr, body_end = nullptr;
  ~Context() {
    cudaSetDevice(device);
    cudaFree(table); cudaFree(table_offsets); cudaFree(table_lengths); cudaFree(unsafe);
    cudaFree(numbers); cudaFree(number_lengths); cudaFree(selected);
    cudaFree(top_counts); cudaFree(fragment_starts); cudaFree(row_lengths);
    cudaFree(row_offsets); cudaFree(status); cudaFree(output);
    cudaFreeHost(host_output); cudaFreeHost(host_offsets); cudaFreeHost(host_status);
    if (kernel_start) cudaEventDestroy(kernel_start);
    if (kernel_end) cudaEventDestroy(kernel_end);
    if (meta_start) cudaEventDestroy(meta_start);
    if (meta_end) cudaEventDestroy(meta_end);
    if (body_start) cudaEventDestroy(body_start);
    if (body_end) cudaEventDestroy(body_end);
  }
};

template <class T> void alloc_device(Context *c, T **ptr, uint64_t count) {
  check(cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)));
  c->device_bytes += count * sizeof(T);
}
template <class T> void alloc_host(Context *c, T **ptr, uint64_t count) {
  check(cudaMallocHost(reinterpret_cast<void **>(ptr), count * sizeof(T)));
  c->pinned_bytes += count * sizeof(T);
}

__device__ int decimal_u32(uint32_t n, char *dst, int pad = 0) {
  char reversed[10];
  int count = 0;
  do { reversed[count++] = char('0' + n % 10); n /= 10; } while (n);
  while (count < pad) reversed[count++] = '0';
  for (int j = 0; j < count; ++j) dst[j] = reversed[count - j - 1];
  return count;
}

// Exact binary32 -> decimal. No float rounding, printf, lookup by probability,
// or reduced precision. The generated JSON number parses to the original
// binary32 promoted to binary64. JSON byte spelling can differ from native.
__device__ int exact_float(float input, char *dst) {
  if (input < -9999.0f) input = -9999.0f;
  uint32_t bits = __float_as_uint(input);
  uint32_t exponent = (bits >> 23) & 255U;
  if (exponent == 255U) return -1;  // NaN and +inf require the compatibility path
  bool negative = (bits >> 31) != 0;
  uint32_t mantissa = bits & 0x7fffffU;
  int pos = 0;
  if (negative) dst[pos++] = '-';
  if (exponent == 0 && mantissa == 0) {
    dst[pos++] = '0'; dst[pos++] = '.'; dst[pos++] = '0';
    return pos;
  }
  int exp2 = exponent ? int(exponent) - 127 - 23 : -149;
  if (exponent) mantissa |= 0x800000U;
  while (exp2 < 0 && (mantissa & 1U) == 0) { mantissa >>= 1; ++exp2; }
  uint32_t limbs[MAX_LIMBS];
  limbs[0] = mantissa;
  int nlimbs = 1;
  int steps = exp2 < 0 ? -exp2 : exp2;
  uint32_t multiplier = exp2 < 0 ? 5U : 2U;
  for (int step = 0; step < steps; ++step) {
    uint64_t carry = 0;
    for (int j = 0; j < nlimbs; ++j) {
      uint64_t value = uint64_t(limbs[j]) * multiplier + carry;
      limbs[j] = uint32_t(value % BASE);
      carry = value / BASE;
    }
    if (carry) limbs[nlimbs++] = uint32_t(carry);
  }
  char digits[180];
  int ndigits = decimal_u32(limbs[nlimbs - 1], digits);
  for (int j = nlimbs - 2; j >= 0; --j)
    ndigits += decimal_u32(limbs[j], digits + ndigits, 9);
  if (exp2 >= 0) {
    for (int j = 0; j < ndigits; ++j) dst[pos++] = digits[j];
    dst[pos++] = '.'; dst[pos++] = '0';
  } else {
    int scale = -exp2;
    int integer_digits = ndigits - scale;
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

template <class ID>
__global__ void prepare(const ID *ids, const float *probs, int rows, int cols, int top_k,
                        uint64_t vocab, const uint8_t *unsafe,
                        const uint32_t *table_lengths, char *numbers,
                        uint32_t *number_lengths, int32_t *selected,
                        int32_t *top_counts, uint32_t *fragment_starts,
                        uint32_t *row_lengths, uint8_t *status) {
  int row = int(blockIdx.x), lane = int(threadIdx.x);
  if (row >= rows) return;
  __shared__ uint8_t entry_status[MAX_COLS];
  __shared__ int64_t row_ids[MAX_COLS];
  for (int j = lane; j < cols; j += int(blockDim.x)) {
    int64_t id = int64_t(ids[row * cols + j]);
    row_ids[j] = id;
    int idx = row * MAX_COLS + j;
    uint8_t result = 0;
    int length = 0;
    if (id < 0 || uint64_t(id) >= vocab) result = 2;
    else if (unsafe[id]) result = 1;
    else {
      length = exact_float(probs[row * cols + j], numbers + idx * NUM_BYTES);
      if (length < 0) { result = 3; length = 0; }
    }
    number_lengths[idx] = uint32_t(length);
    entry_status[j] = result;
  }
  __syncthreads();
  if (lane != 0) return;
  uint8_t result = 0;
  for (int j = 0; j < cols; ++j) if (entry_status[j]) { result = entry_status[j]; break; }
  status[row] = result;
  row_lengths[row] = 0;
  top_counts[row] = 0;
  if (result) return;

  int unique_last[MAX_COLS];
  int unique_count = 0;
  for (int j = 0; j < cols; ++j) {
    bool first = true;
    for (int i = 0; i < j; ++i) if (row_ids[i] == row_ids[j]) { first = false; break; }
    if (!first) continue;
    int last = j;
    for (int i = j + 1; i < cols; ++i) if (row_ids[i] == row_ids[j]) last = i;
    unique_last[unique_count++] = last;
  }
  int count = unique_count < top_k ? unique_count : top_k;
  top_counts[row] = count;
  int base = row * MAX_COLS;
  selected[base] = unique_last[0];  // sampled token, with last duplicate's value
  fragment_starts[base] = 0;
  int sampled = unique_last[0];
  uint64_t sample_id = uint64_t(row_ids[sampled]);
  uint32_t offset = table_lengths[sample_id * 2] + table_lengths[sample_id * 2 + 1]
                    + number_lengths[base + sampled] + TOP_INTRO_LEN;
  for (int j = 0; j < count; ++j) {
    int source = unique_last[j];
    uint64_t id = uint64_t(row_ids[source]);
    selected[base + j + 1] = source;
    fragment_starts[base + j + 1] = offset;
    offset += table_lengths[id * 2] + table_lengths[id * 2 + 1]
              + number_lengths[base + source] + 1;  // closing object brace
    if (j + 1 < count) ++offset;  // separator
  }
  row_lengths[row] = offset + 2;  // ]}
}

__global__ void prefix_rows(const uint32_t *lengths, uint64_t *offsets, int rows) {
  if (blockIdx.x || threadIdx.x) return;
  uint64_t total = 0;
  for (int row = 0; row < rows; ++row) { offsets[row] = total; total += lengths[row]; }
  offsets[rows] = total;
}

__device__ uint32_t copy_bytes(char *out, uint32_t pos, const char *src, uint32_t count) {
  for (uint32_t j = 0; j < count; ++j) out[pos + j] = src[j];
  return pos + count;
}

template <class ID>
__global__ void write_rows(const ID *ids, int rows, int cols, const char *table,
                           const uint64_t *table_offsets, const uint32_t *table_lengths,
                           const char *numbers, const uint32_t *number_lengths,
                           const int32_t *selected, const int32_t *top_counts,
                           const uint32_t *fragment_starts, const uint32_t *row_lengths,
                           const uint64_t *row_offsets, const uint8_t *status,
                           char *output, uint64_t output_capacity) {
  int row = int(blockIdx.x), lane = int(threadIdx.x);
  if (row >= rows || status[row] || row_offsets[rows] > output_capacity) return;
  int count = top_counts[row], base = row * MAX_COLS;
  char *out = output + row_offsets[row];
  for (int entry = lane; entry <= count; entry += int(blockDim.x)) {
    int source = selected[base + entry];
    uint64_t id = uint64_t(ids[row * cols + source]);
    uint32_t pos = fragment_starts[base + entry];
    pos = copy_bytes(out, pos, table + table_offsets[id * 2], table_lengths[id * 2]);
    pos = copy_bytes(out, pos, numbers + (base + source) * NUM_BYTES,
                     number_lengths[base + source]);
    pos = copy_bytes(out, pos, table + table_offsets[id * 2 + 1], table_lengths[id * 2 + 1]);
    if (entry == 0) {
      const char intro[] = ",\"top_logprobs\":[";
      pos = copy_bytes(out, pos, intro, TOP_INTRO_LEN);
    } else {
      out[pos++] = '}';
      if (entry < count) out[pos++] = ',';
    }
  }
  if (lane == 0) { out[row_lengths[row] - 2] = ']'; out[row_lengths[row] - 1] = '}'; }
}
}  // namespace

extern "C" {

const char *f_gpu_error() { return last_error.c_str(); }

void *f_gpu_create(const char *blob, uint64_t blob_bytes, const uint64_t *offsets,
                   const uint32_t *lengths, const uint8_t *unsafe,
                   uint64_t vocab, int max_rows, int device) {
  Context *c = nullptr;
  try {
    if (!vocab || max_rows < 1 || max_rows > 65536)
      throw std::runtime_error("invalid vocabulary or row capacity");
    check(cudaSetDevice(device));
    c = new Context(); c->device = device; c->vocab = vocab; c->max_rows = max_rows;
    uint64_t max_piece = 0;
    for (uint64_t id = 0; id < vocab; ++id) {
      for (int side = 0; side < 2; ++side)
        if (offsets[id * 2 + side] + lengths[id * 2 + side] > blob_bytes)
          throw std::runtime_error("token table offset outside blob");
      max_piece = std::max(max_piece, uint64_t(lengths[id * 2]) + lengths[id * 2 + 1]);
    }
    uint64_t max_row_bytes = (max_piece + NUM_BYTES + 2) * MAX_COLS + TOP_INTRO_LEN + 2;
    c->output_capacity = uint64_t(max_rows) * max_row_bytes;
    if (max_row_bytes > UINT32_MAX || c->output_capacity > (uint64_t(1) << 34))
      throw std::runtime_error("token table requires excessive output capacity");
    alloc_device(c, &c->table, std::max(uint64_t(1), blob_bytes));
    alloc_device(c, &c->table_offsets, vocab * 2);
    alloc_device(c, &c->table_lengths, vocab * 2);
    alloc_device(c, &c->unsafe, vocab);
    check(cudaMemcpy(c->table, blob, blob_bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->table_offsets, offsets, vocab * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->table_lengths, lengths, vocab * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(c->unsafe, unsafe, vocab, cudaMemcpyHostToDevice));
    uint64_t entries = uint64_t(max_rows) * MAX_COLS;
    alloc_device(c, &c->numbers, entries * NUM_BYTES);
    alloc_device(c, &c->number_lengths, entries);
    alloc_device(c, &c->selected, entries);
    alloc_device(c, &c->fragment_starts, entries);
    alloc_device(c, &c->top_counts, max_rows);
    alloc_device(c, &c->row_lengths, max_rows);
    alloc_device(c, &c->row_offsets, max_rows + 1);
    alloc_device(c, &c->status, max_rows);
    alloc_device(c, &c->output, c->output_capacity);
    alloc_host(c, &c->host_output, c->output_capacity);
    alloc_host(c, &c->host_offsets, max_rows + 1);
    alloc_host(c, &c->host_status, max_rows);
    check(cudaEventCreate(&c->kernel_start)); check(cudaEventCreate(&c->kernel_end));
    check(cudaEventCreate(&c->meta_start)); check(cudaEventCreate(&c->meta_end));
    check(cudaEventCreate(&c->body_start)); check(cudaEventCreate(&c->body_end));
    return c;
  } catch (const std::exception &e) { last_error = e.what(); delete c; return nullptr; }
}

void f_gpu_destroy(void *handle) { delete static_cast<Context *>(handle); }

uint64_t f_gpu_device_bytes(void *handle) { return static_cast<Context *>(handle)->device_bytes; }
uint64_t f_gpu_pinned_bytes(void *handle) { return static_cast<Context *>(handle)->pinned_bytes; }

int f_gpu_encode(void *handle, const void *ids, int id_bytes, const float *probs,
                 int rows, int cols, int top_k, uint64_t stream_ptr,
                 const char **packed, const uint64_t **offsets, const uint8_t **statuses,
                 uint64_t *packed_bytes, float *kernel_ms, float *d2h_ms) {
  try {
    Context *c = static_cast<Context *>(handle);
    if (!c || !ids || !probs || rows < 1 || rows > c->max_rows || cols < 1 ||
        cols > MAX_COLS || top_k < 0 || top_k >= MAX_COLS || (id_bytes != 4 && id_bytes != 8))
      throw std::runtime_error("invalid encode inputs or insufficient reserved row capacity");
    check(cudaSetDevice(c->device));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    check(cudaEventRecord(c->kernel_start, stream));
    if (id_bytes == 4)
      prepare<<<rows, 32, 0, stream>>>(static_cast<const int32_t *>(ids), probs, rows, cols, top_k,
        c->vocab, c->unsafe, c->table_lengths, c->numbers, c->number_lengths, c->selected,
        c->top_counts, c->fragment_starts, c->row_lengths, c->status);
    else
      prepare<<<rows, 32, 0, stream>>>(static_cast<const int64_t *>(ids), probs, rows, cols, top_k,
        c->vocab, c->unsafe, c->table_lengths, c->numbers, c->number_lengths, c->selected,
        c->top_counts, c->fragment_starts, c->row_lengths, c->status);
    prefix_rows<<<1, 1, 0, stream>>>(c->row_lengths, c->row_offsets, rows);
    if (id_bytes == 4)
      write_rows<<<rows, 32, 0, stream>>>(static_cast<const int32_t *>(ids), rows, cols, c->table,
        c->table_offsets, c->table_lengths, c->numbers, c->number_lengths, c->selected,
        c->top_counts, c->fragment_starts, c->row_lengths, c->row_offsets, c->status,
        c->output, c->output_capacity);
    else
      write_rows<<<rows, 32, 0, stream>>>(static_cast<const int64_t *>(ids), rows, cols, c->table,
        c->table_offsets, c->table_lengths, c->numbers, c->number_lengths, c->selected,
        c->top_counts, c->fragment_starts, c->row_lengths, c->row_offsets, c->status,
        c->output, c->output_capacity);
    check(cudaGetLastError());
    check(cudaEventRecord(c->kernel_end, stream));
    check(cudaEventRecord(c->meta_start, stream));
    check(cudaMemcpyAsync(c->host_offsets, c->row_offsets, (rows + 1) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(c->host_status, c->status, rows, cudaMemcpyDeviceToHost, stream));
    check(cudaEventRecord(c->meta_end, stream));
    check(cudaEventSynchronize(c->meta_end));
    uint64_t count = c->host_offsets[rows];
    if (count > c->output_capacity) throw std::runtime_error("output capacity exceeded");
    check(cudaEventRecord(c->body_start, stream));
    if (count) check(cudaMemcpyAsync(c->host_output, c->output, count, cudaMemcpyDeviceToHost, stream));
    check(cudaEventRecord(c->body_end, stream));
    check(cudaEventSynchronize(c->body_end));
    float meta_ms = 0, body_ms = 0;
    check(cudaEventElapsedTime(kernel_ms, c->kernel_start, c->kernel_end));
    check(cudaEventElapsedTime(&meta_ms, c->meta_start, c->meta_end));
    check(cudaEventElapsedTime(&body_ms, c->body_start, c->body_end));
    *d2h_ms = meta_ms + body_ms;
    *packed = c->host_output; *offsets = c->host_offsets; *statuses = c->host_status;
    *packed_bytes = count;
    return 0;
  } catch (const std::exception &e) { last_error = e.what(); return 1; }
}
}  // extern C
