// C++17, no Python or model dependencies.  See cpu_encoder.py for the ABI.
// This changes representation only: each input row is one already-ready position.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Encoder {
    std::vector<char> table;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> lengths;
    std::vector<uint8_t> unsafe;
    std::vector<char> output;
    std::vector<uint64_t> row_offsets;
    std::vector<uint8_t> status;
    std::string error;
    size_t vocab;

    Encoder(const char* blob, size_t size, const uint64_t* off,
            const uint32_t* len, const uint8_t* bad, size_t n)
        : offsets(off, off + 2 * n), lengths(len, len + 2 * n),
          unsafe(bad, bad + n), vocab(n) {
        if (size) table.assign(blob, blob + size);
        for (size_t i = 0; i < 2 * n; ++i) {
            if (offsets[i] > size || lengths[i] > size - offsets[i])
                throw std::invalid_argument("token table span out of bounds");
        }
    }
    void literal(const char* text, size_t n) {
        output.insert(output.end(), text, text + n);
    }
    template<size_t N> void literal(const char (&text)[N]) {
        literal(text, N - 1);
    }
    void piece(size_t id, size_t part) {
        size_t i = 2 * id + part;
        if (lengths[i]) {
            const char* begin = table.data() + offsets[i];
            output.insert(output.end(), begin, begin + lengths[i]);
        }
    }
    void entry(size_t id, float logprob) {
        piece(id, 0);
        // Python .tolist() first promotes float32 to double.  Emit a shortest
        // double representation, NOT a shortest float representation: parsed
        // JSON values must equal the native float32-promoted values exactly.
        const double value = std::max(static_cast<double>(logprob), -9999.0);
        char buffer[64];
        auto r = std::to_chars(buffer, buffer + sizeof(buffer), value,
                               std::chars_format::general);
        if (r.ec != std::errc()) throw std::runtime_error("to_chars failed");
        literal(buffer, static_cast<size_t>(r.ptr - buffer));
        // Preserve floating JSON type and the sign of -0.0 after parsing.
        if (std::find(buffer, r.ptr, '.') == r.ptr &&
            std::find(buffer, r.ptr, 'e') == r.ptr &&
            std::find(buffer, r.ptr, 'E') == r.ptr) literal(".0");
        piece(id, 1);
    }

    void encode(const int64_t* ids, const float* lp, size_t rows,
                size_t columns, size_t top_k) {
        output.clear();
        row_offsets.resize(rows + 1);
        status.assign(rows, 0);
        std::vector<size_t> first;
        std::vector<size_t> last;
        first.reserve(columns);
        last.reserve(columns);
        for (size_t row = 0; row < rows; ++row) {
            row_offsets[row] = output.size();
            const int64_t* ri = ids + row * columns;
            const float* rp = lp + row * columns;
            // Validate before writing. Unsafe rows receive zero output bytes;
            // the caller must invoke the native context-aware fallback.
            for (size_t j = 0; j < columns; ++j) {
                if (ri[j] < 0 || static_cast<uint64_t>(ri[j]) >= vocab) {
                    status[row] = 2;
                    break;
                }
                if (std::isnan(rp[j]) || rp[j] ==
                    std::numeric_limits<float>::infinity()) {
                    status[row] = 3;
                    break;
                }
                if (unsafe[static_cast<size_t>(ri[j])]) status[row] = 1;
            }
            if (status[row]) continue;
            first.clear();
            last.clear();
            // K=20 is intentionally small. O(K^2) stack/cache-local comparisons
            // avoid constructing a hash map and preserve Python dict order.
            for (size_t j = 0; j < columns; ++j) {
                size_t found = first.size();
                for (size_t u = 0; u < first.size(); ++u) {
                    if (ri[first[u]] == ri[j]) { found = u; break; }
                }
                if (found == first.size()) {
                    first.push_back(j);
                    last.push_back(j);
                } else {
                    last[found] = j;
                }
            }
            entry(static_cast<size_t>(ri[0]), rp[last[0]]);
            literal(",\"top_logprobs\":[");
            size_t n = std::min(top_k, first.size());
            for (size_t u = 0; u < n; ++u) {
                if (u) literal(",");
                entry(static_cast<size_t>(ri[first[u]]), rp[last[u]]);
                literal("}");
            }
            literal("]}");
        }
        row_offsets[rows] = output.size();
    }
};
thread_local std::string creation_error;
}  // namespace

extern "C" {
uint32_t f_cpu_encoder_abi() { return 1; }

void* f_cpu_encoder_create(const char* blob, uint64_t blob_size,
                           const uint64_t* offsets, const uint32_t* lengths,
                           const uint8_t* unsafe, uint64_t vocab_size) {
    try {
        creation_error.clear();
        if (!offsets || !lengths || !unsafe || !vocab_size ||
            (!blob && blob_size) || vocab_size > SIZE_MAX / 2)
            throw std::invalid_argument("invalid token table arguments");
        return new Encoder(blob, blob_size, offsets, lengths, unsafe, vocab_size);
    } catch (const std::exception& e) {
        creation_error = e.what();
        return nullptr;
    } catch (...) {
        creation_error = "unknown encoder creation error";
        return nullptr;
    }
}

// Returned spans belong to handle and remain valid until the next encode or
// destruction. One handle must not be entered concurrently.  No token strings,
// per-candidate objects or dictionaries are constructed in this path.
int f_cpu_encoder_encode(void* handle, const int64_t* ids, const float* logprobs,
                          uint64_t rows, uint64_t columns, uint64_t top_k,
                          const char** packed, uint64_t* packed_size,
                          const uint64_t** offsets, const uint8_t** status) {
    if (!handle) return -1;
    Encoder& e = *static_cast<Encoder*>(handle);
    try {
        e.error.clear();
        if (!packed || !packed_size || !offsets || !status ||
            (rows && (!ids || !logprobs)) || !columns || columns > 256 ||
            top_k > columns || rows > (SIZE_MAX - 1) / columns)
            throw std::invalid_argument("invalid input shape or pointers");
        e.encode(ids, logprobs, rows, columns, top_k);
        *packed = e.output.data();
        *packed_size = e.output.size();
        *offsets = e.row_offsets.data();
        *status = e.status.data();
        return 0;
    } catch (const std::exception& ex) {
        e.error = ex.what();
        return -1;
    } catch (...) {
        e.error = "unknown encode error";
        return -1;
    }
}

const char* f_cpu_encoder_error(void* handle) {
    return handle ? static_cast<Encoder*>(handle)->error.c_str()
                  : creation_error.c_str();
}
void f_cpu_encoder_destroy(void* handle) { delete static_cast<Encoder*>(handle); }
}
