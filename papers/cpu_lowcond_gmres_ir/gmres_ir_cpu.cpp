// CPU dense mixed-precision GMRES-IR validation runner for RL-selected actions.
//
// Stage mapping follows the dense_test Python solver convention:
//   uf: factorization/preconditioner storage precision
//   ug: GMRES inner arithmetic precision
//   u : working/update precision
//   ur: residual precision
//
// The implementation is standalone C++17.  It uses dense partial-pivoting LU for
// the uf preconditioner/factor stage, fp16 storage plus F16C conversions when
// available, and AVX-512 row kernels when the compiler target supports them.
// Final acceptance is the user-facing forward-error threshold; backward error is
// still reported as an independent diagnostic.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#define RLCPU_X86 1
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr char kMagic[8] = {'R', 'L', 'C', 'P', 'U', '1', '\0', '\0'};

struct Problem {
    std::string file;
    uint32_t n = 0;
    uint32_t split_id = 0;
    double target_cond = 1.0;
    std::vector<double> A;
    std::vector<double> b;
    std::vector<double> x_true;
};

struct Options {
    std::string manifest;
    std::string data_dir;
    std::string output;
    std::string split = "test";
    std::string uf = "fp64";
    std::string ug = "fp64";
    std::string u = "fp64";
    std::string ur = "fp64";
    int repeats = 3;
    int outer_max_iters = 10;
    int gmres_max_iters = 30;
    int gmres_cycles = 1;
    double gmres_tolerance = 1e-4;
    double tolerance = 1e-6;
    double backward_tolerance = 1e-14;
};

struct Row {
    std::string file;
    std::string split;
    uint32_t n = 0;
    double target_cond = 1.0;
    std::string uf, ug, u, ur;
    std::string status = "ok";
    int outer_iters = 0;
    int gmres_iterations = 0;
    double avg_ms = 0.0;
    double rel_error = 0.0;
    double backward_error = 0.0;
    double a_norm_inf = 0.0;
    double memory_bytes = 0.0;
};

static uint16_t f32_to_f16_portable(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float f16_to_f32_portable(uint16_t half) {
    uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t mant = half & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t f32_to_f16(float value) {
#if defined(RLCPU_X86) && defined(__F16C__)
    return static_cast<uint16_t>(_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
    return f32_to_f16_portable(value);
#endif
}

static float f16_to_f32(uint16_t value) {
#if defined(RLCPU_X86) && defined(__F16C__)
    return _cvtsh_ss(value);
#else
    return f16_to_f32_portable(value);
#endif
}

static bool valid_precision(const std::string& p) {
    return p == "fp16" || p == "fp32" || p == "fp64";
}

static double quantize(double value, const std::string& precision) {
    if (precision == "fp64") return value;
    float as_float = static_cast<float>(value);
    if (precision == "fp32") return static_cast<double>(as_float);
    return static_cast<double>(f16_to_f32(f32_to_f16(as_float)));
}

static void quantize_vector(std::vector<double>& values, const std::string& precision) {
    if (precision == "fp64") return;
    for (double& value : values) value = quantize(value, precision);
}

static double storage_bytes(const std::string& precision) {
    if (precision == "fp16") return 2.0;
    if (precision == "fp32") return 4.0;
    return 8.0;
}

static bool read_problem(const std::string& path, Problem& p) {
    std::ifstream handle(path, std::ios::binary);
    if (!handle) return false;
    char magic[8];
    handle.read(magic, 8);
    if (!handle || std::memcmp(magic, kMagic, 8) != 0) return false;
    handle.read(reinterpret_cast<char*>(&p.n), sizeof(p.n));
    handle.read(reinterpret_cast<char*>(&p.split_id), sizeof(p.split_id));
    handle.read(reinterpret_cast<char*>(&p.target_cond), sizeof(p.target_cond));
    const size_t n = p.n;
    p.A.resize(n * n);
    p.b.resize(n);
    p.x_true.resize(n);
    handle.read(reinterpret_cast<char*>(p.A.data()), n * n * sizeof(double));
    handle.read(reinterpret_cast<char*>(p.b.data()), n * sizeof(double));
    handle.read(reinterpret_cast<char*>(p.x_true.data()), n * sizeof(double));
    return static_cast<bool>(handle);
}

static std::vector<std::map<std::string, std::string>> read_manifest(const std::string& path) {
    std::ifstream handle(path);
    if (!handle) throw std::runtime_error("cannot open manifest: " + path);
    std::string line;
    if (!std::getline(handle, line)) return {};
    std::vector<std::string> headers;
    std::stringstream hs(line);
    std::string field;
    while (std::getline(hs, field, ',')) headers.push_back(field);
    std::vector<std::map<std::string, std::string>> rows;
    while (std::getline(handle, line)) {
        if (line.empty()) continue;
        std::stringstream ls(line);
        std::map<std::string, std::string> row;
        for (const auto& header : headers) {
            if (!std::getline(ls, field, ',')) field.clear();
            row[header] = field;
        }
        rows.push_back(row);
    }
    return rows;
}

static double dot64(const double* a, const double* b, size_t n) {
#if defined(RLCPU_X86) && defined(__AVX512F__)
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d va = _mm512_loadu_pd(a + i);
        const __m512d vb = _mm512_loadu_pd(b + i);
        acc = _mm512_fmadd_pd(va, vb, acc);
    }
    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

static float dot32(const float* a, const float* b, size_t n) {
#if defined(RLCPU_X86) && defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, acc);
    float sum = 0.0f;
    for (float lane : lanes) sum += lane;
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

static double dot32_as64(const float* a, const double* b, size_t n) {
#if defined(RLCPU_X86) && defined(__AVX512F__)
    __m512d acc = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 af = _mm256_loadu_ps(a + i);
        const __m512d ad = _mm512_cvtps_pd(af);
        const __m512d bd = _mm512_loadu_pd(b + i);
        acc = _mm512_fmadd_pd(ad, bd, acc);
    }
    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
#endif
}

static double half_dot_as64(const uint16_t* a, const double* b, size_t n) {
#if defined(RLCPU_X86) && defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i hv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m512 hf = _mm512_cvtph_ps(hv);
        const __m256 flo = _mm512_castps512_ps256(hf);
        const __m256 fhi = _mm512_extractf32x8_ps(hf, 1);
        const __m512d dlo = _mm512_cvtps_pd(flo);
        const __m512d dhi = _mm512_cvtps_pd(fhi);
        const __m512d blo = _mm512_loadu_pd(b + i);
        const __m512d bhi = _mm512_loadu_pd(b + i + 8);
        acc0 = _mm512_fmadd_pd(dlo, blo, acc0);
        acc1 = _mm512_fmadd_pd(dhi, bhi, acc1);
    }
    alignas(64) double l0[8], l1[8];
    _mm512_store_pd(l0, acc0);
    _mm512_store_pd(l1, acc1);
    double sum = 0.0;
    for (int k = 0; k < 8; ++k) sum += l0[k] + l1[k];
    for (; i < n; ++i) sum += static_cast<double>(f16_to_f32(a[i])) * b[i];
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(f16_to_f32(a[i])) * b[i];
    return sum;
#endif
}

static double dot_quantized_ptr(const std::vector<double>& a, const double* b, size_t n, const std::string& precision) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum = quantize(sum + quantize(a[i] * b[i], precision), precision);
    return sum;
}

static double norm2(const std::vector<double>& x) {
    return std::sqrt(std::max(0.0, dot64(x.data(), x.data(), x.size())));
}

static double inf_norm(const std::vector<double>& x) {
    double value = 0.0;
    for (double item : x) value = std::max(value, std::abs(item));
    return value;
}

static double matrix_inf_norm(const std::vector<double>& A, size_t n) {
    double value = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < n; ++j) row_sum += std::abs(A[i * n + j]);
        value = std::max(value, row_sum);
    }
    return value;
}

struct PrecisionMatrix {
    std::string precision;
    size_t n = 0;
    std::vector<double> A64;
    std::vector<float> A32;
    std::vector<uint16_t> A16;

    PrecisionMatrix(const Problem& p, std::string mode) : precision(std::move(mode)), n(p.n) {
        if (precision == "fp64") {
            A64 = p.A;
        } else if (precision == "fp32") {
            A32.resize(p.A.size());
            for (size_t i = 0; i < p.A.size(); ++i) A32[i] = static_cast<float>(p.A[i]);
        } else if (precision == "fp16") {
            A16.resize(p.A.size());
            for (size_t i = 0; i < p.A.size(); ++i) A16[i] = f32_to_f16(static_cast<float>(p.A[i]));
        } else {
            throw std::runtime_error("unknown precision: " + precision);
        }
    }

    double at(size_t i, size_t j) const {
        const size_t index = i * n + j;
        if (precision == "fp64") return A64[index];
        if (precision == "fp32") return static_cast<double>(A32[index]);
        return static_cast<double>(f16_to_f32(A16[index]));
    }

    void matvec(const std::vector<double>& x, std::vector<double>& y, const std::string& compute_precision) const {
        y.assign(n, 0.0);
        if (compute_precision == "fp64") {
            for (size_t i = 0; i < n; ++i) {
                if (precision == "fp64") y[i] = dot64(&A64[i * n], x.data(), n);
                else if (precision == "fp32") y[i] = dot32_as64(&A32[i * n], x.data(), n);
                else y[i] = half_dot_as64(&A16[i * n], x.data(), n);
            }
            return;
        }
        std::vector<float> xf(n);
        for (size_t i = 0; i < n; ++i) {
            const float value = static_cast<float>(x[i]);
            xf[i] = compute_precision == "fp16" ? f16_to_f32(f32_to_f16(value)) : value;
        }
        for (size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            if (precision == "fp32") {
                sum = dot32(&A32[i * n], xf.data(), n);
            } else {
                for (size_t j = 0; j < n; ++j) sum += static_cast<float>(at(i, j)) * xf[j];
            }
            y[i] = quantize(static_cast<double>(sum), compute_precision);
        }
    }
};

struct LUPreconditioner {
    std::string precision;
    size_t n = 0;
    std::vector<int> piv;
    std::vector<double> LU64;
    std::vector<float> LU32;
    std::vector<uint16_t> LU16;

    LUPreconditioner(const Problem& p, const std::string& mode) : precision(mode), n(p.n) {
        bool ok = false;
        if (precision == "fp64") ok = factor64(p);
        else if (precision == "fp32") ok = factor32(p);
        else if (precision == "fp16") ok = factor16(p);
        else throw std::runtime_error("unknown precision: " + precision);
        if (!ok) throw std::runtime_error("dense LU factorization failed for uf=" + precision);
    }

    bool factor64(const Problem& p) {
        LU64 = p.A;
        piv.resize(n);
        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            double amax = std::abs(LU64[k * n + k]);
            for (size_t i = k + 1; i < n; ++i) {
                const double value = std::abs(LU64[i * n + k]);
                if (value > amax) { amax = value; pivot_row = i; }
            }
            piv[k] = static_cast<int>(pivot_row);
            if (!(amax > 0.0) || !std::isfinite(amax)) return false;
            if (pivot_row != k) {
                for (size_t j = 0; j < n; ++j) std::swap(LU64[k * n + j], LU64[pivot_row * n + j]);
            }
            const double pivot = LU64[k * n + k];
            for (size_t i = k + 1; i < n; ++i) {
                double* rowi = &LU64[i * n];
                const double* rowk = &LU64[k * n];
                const double lik = rowi[k] / pivot;
                rowi[k] = lik;
#if defined(RLCPU_X86) && defined(__AVX512F__)
                const __m512d vl = _mm512_set1_pd(lik);
                size_t j = k + 1;
                for (; j + 8 <= n; j += 8) {
                    const __m512d vi = _mm512_loadu_pd(rowi + j);
                    const __m512d vk = _mm512_loadu_pd(rowk + j);
                    _mm512_storeu_pd(rowi + j, _mm512_fnmadd_pd(vl, vk, vi));
                }
                for (; j < n; ++j) rowi[j] -= lik * rowk[j];
#else
                for (size_t j = k + 1; j < n; ++j) rowi[j] -= lik * rowk[j];
#endif
            }
        }
        return true;
    }

    bool factor32(const Problem& p) {
        LU32.resize(p.A.size());
        for (size_t i = 0; i < p.A.size(); ++i) LU32[i] = static_cast<float>(p.A[i]);
        piv.resize(n);
        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            float amax = std::abs(LU32[k * n + k]);
            for (size_t i = k + 1; i < n; ++i) {
                const float value = std::abs(LU32[i * n + k]);
                if (value > amax) { amax = value; pivot_row = i; }
            }
            piv[k] = static_cast<int>(pivot_row);
            if (!(amax > 0.0f) || !std::isfinite(amax)) return false;
            if (pivot_row != k) {
                for (size_t j = 0; j < n; ++j) std::swap(LU32[k * n + j], LU32[pivot_row * n + j]);
            }
            const float pivot = LU32[k * n + k];
            for (size_t i = k + 1; i < n; ++i) {
                float* rowi = &LU32[i * n];
                const float* rowk = &LU32[k * n];
                const float lik = rowi[k] / pivot;
                rowi[k] = lik;
#if defined(RLCPU_X86) && defined(__AVX512F__)
                const __m512 vl = _mm512_set1_ps(lik);
                size_t j = k + 1;
                for (; j + 16 <= n; j += 16) {
                    const __m512 vi = _mm512_loadu_ps(rowi + j);
                    const __m512 vk = _mm512_loadu_ps(rowk + j);
                    _mm512_storeu_ps(rowi + j, _mm512_fnmadd_ps(vl, vk, vi));
                }
                for (; j < n; ++j) rowi[j] -= lik * rowk[j];
#else
                for (size_t j = k + 1; j < n; ++j) rowi[j] -= lik * rowk[j];
#endif
            }
        }
        return true;
    }

    bool factor16(const Problem& p) {
        LU16.resize(p.A.size());
        for (size_t i = 0; i < p.A.size(); ++i) LU16[i] = f32_to_f16(static_cast<float>(p.A[i]));
        piv.resize(n);
        for (size_t k = 0; k < n; ++k) {
            size_t pivot_row = k;
            float amax = std::abs(f16_to_f32(LU16[k * n + k]));
            for (size_t i = k + 1; i < n; ++i) {
                const float value = std::abs(f16_to_f32(LU16[i * n + k]));
                if (value > amax) { amax = value; pivot_row = i; }
            }
            piv[k] = static_cast<int>(pivot_row);
            if (!(amax > 0.0f) || !std::isfinite(amax)) return false;
            if (pivot_row != k) {
                for (size_t j = 0; j < n; ++j) std::swap(LU16[k * n + j], LU16[pivot_row * n + j]);
            }
            const float pivot = f16_to_f32(LU16[k * n + k]);
            for (size_t i = k + 1; i < n; ++i) {
                uint16_t* rowi = &LU16[i * n];
                const uint16_t* rowk = &LU16[k * n];
                const float lik = f16_to_f32(f32_to_f16(f16_to_f32(rowi[k]) / pivot));
                rowi[k] = f32_to_f16(lik);
#if defined(RLCPU_X86) && defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
                const __m512 vl = _mm512_set1_ps(lik);
                size_t j = k + 1;
                for (; j + 16 <= n; j += 16) {
                    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowi + j));
                    const __m256i hk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowk + j));
                    const __m512 ai = _mm512_cvtph_ps(hi);
                    const __m512 ak = _mm512_cvtph_ps(hk);
                    const __m512 y = _mm512_fnmadd_ps(vl, ak, ai);
                    const __m256i hy = _mm512_cvtps_ph(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowi + j), hy);
                }
                for (; j < n; ++j) rowi[j] = f32_to_f16(std::fma(-lik, f16_to_f32(rowk[j]), f16_to_f32(rowi[j])));
#else
                for (size_t j = k + 1; j < n; ++j) rowi[j] = f32_to_f16(std::fma(-lik, f16_to_f32(rowk[j]), f16_to_f32(rowi[j])));
#endif
            }
        }
        return true;
    }

    bool apply(const std::vector<double>& rhs, std::vector<double>& y, const std::string& output_precision) const {
        y = rhs;
        for (size_t k = 0; k < n; ++k) {
            const size_t pk = static_cast<size_t>(piv[k]);
            if (pk != k) std::swap(y[k], y[pk]);
        }
        for (size_t i = 0; i < n; ++i) {
            double sum = y[i];
            if (precision == "fp64") sum -= dot64(&LU64[i * n], y.data(), i);
            else if (precision == "fp32") sum -= dot32_as64(&LU32[i * n], y.data(), i);
            else sum -= half_dot_as64(&LU16[i * n], y.data(), i);
            y[i] = sum;
        }
        for (size_t ii = 0; ii < n; ++ii) {
            const size_t i = n - 1 - ii;
            double sum = y[i];
            if (i + 1 < n) {
                if (precision == "fp64") sum -= dot64(&LU64[i * n + i + 1], y.data() + i + 1, n - i - 1);
                else if (precision == "fp32") sum -= dot32_as64(&LU32[i * n + i + 1], y.data() + i + 1, n - i - 1);
                else sum -= half_dot_as64(&LU16[i * n + i + 1], y.data() + i + 1, n - i - 1);
            }
            double diag = 0.0;
            if (precision == "fp64") diag = LU64[i * n + i];
            else if (precision == "fp32") diag = static_cast<double>(LU32[i * n + i]);
            else diag = static_cast<double>(f16_to_f32(LU16[i * n + i]));
            if (!(std::abs(diag) > 0.0) || !std::isfinite(diag)) return false;
            y[i] = sum / diag;
            if (!std::isfinite(y[i])) return false;
        }
        quantize_vector(y, output_precision);
        return true;
    }
};

static double rel_error(const std::vector<double>& x, const std::vector<double>& ref) {
    std::vector<double> diff(x.size());
    for (size_t i = 0; i < x.size(); ++i) diff[i] = x[i] - ref[i];
    const double denom = std::max(norm2(ref), std::numeric_limits<double>::min());
    return norm2(diff) / denom;
}

static double backward_error(const Problem& p, const std::vector<double>& x, double anorm, double bnorm) {
    const size_t n = p.n;
    std::vector<double> r(n, 0.0);
    for (size_t i = 0; i < n; ++i) r[i] = p.b[i] - dot64(&p.A[i * n], x.data(), n);
    const double denom = anorm * inf_norm(x) + bnorm;
    return denom > 0.0 ? inf_norm(r) / denom : inf_norm(r);
}

static void residual(const Problem& p, const PrecisionMatrix& A_res, const std::vector<double>& x, const std::string& ur, std::vector<double>& r) {
    std::vector<double> ax;
    A_res.matvec(x, ax, ur);
    r.resize(p.n);
    for (size_t i = 0; i < p.n; ++i) r[i] = quantize(quantize(p.b[i], ur) - ax[i], ur);
}

static void givens(double a, double b, double& c, double& s) {
    if (b == 0.0) { c = 1.0; s = 0.0; return; }
    if (std::abs(b) > std::abs(a)) {
        const double t = a / b;
        s = 1.0 / std::sqrt(1.0 + t * t);
        c = t * s;
    } else {
        const double t = b / a;
        c = 1.0 / std::sqrt(1.0 + t * t);
        s = t * c;
    }
}

static bool upper_hessenberg_solve(const std::vector<double>& H, int ld, const std::vector<double>& s, int dim, std::vector<double>& y) {
    y.assign(dim, 0.0);
    for (int i = dim - 1; i >= 0; --i) {
        double value = s[static_cast<size_t>(i)];
        for (int j = i + 1; j < dim; ++j) value -= H[static_cast<size_t>(i) * ld + j] * y[static_cast<size_t>(j)];
        const double diag = H[static_cast<size_t>(i) * ld + i];
        if (!(std::abs(diag) > 0.0) || !std::isfinite(diag)) return false;
        y[static_cast<size_t>(i)] = value / diag;
    }
    return true;
}

static int gmres_step(const PrecisionMatrix& A_work, const LUPreconditioner& preconditioner, const std::vector<double>& r, const std::string& ug, int restart, int max_cycles, double gmres_tol, std::vector<double>& dx) {
    const size_t n = A_work.n;
    const int requested_restart = restart <= 0 ? static_cast<int>(n) : restart;
    const int m = std::max(1, std::min(requested_restart, static_cast<int>(n)));
    dx.assign(n, 0.0);

    std::vector<double> prhs;
    if (!preconditioner.apply(r, prhs, ug)) return 0;
    const double bnrm2 = std::max(norm2(prhs), std::numeric_limits<double>::min());
    if (bnrm2 == 0.0) return 1;

    std::vector<double> V(static_cast<size_t>(m + 1) * n);
    std::vector<double> H(static_cast<size_t>(m + 1) * m, 0.0);
    std::vector<double> cs(m), sn(m), svec(m + 1), y;
    std::vector<double> residual_vec(n), av, w, op_dx;
    int total_used = 0;

    for (int cycle = 0; cycle < std::max(1, max_cycles); ++cycle) {
        if (cycle == 0) {
            residual_vec = prhs;
        } else {
            A_work.matvec(dx, av, ug);
            if (!preconditioner.apply(av, op_dx, ug)) return total_used;
            for (size_t i = 0; i < n; ++i) residual_vec[i] = quantize(prhs[i] - op_dx[i], ug);
        }
        double beta = norm2(residual_vec);
        if (!(beta > 0.0) || !std::isfinite(beta)) break;
        if (beta / bnrm2 <= gmres_tol) break;

        std::fill(H.begin(), H.end(), 0.0);
        std::fill(svec.begin(), svec.end(), 0.0);
        for (size_t i = 0; i < n; ++i) V[i] = quantize(residual_vec[i] / beta, ug);
        svec[0] = beta;

        int used = 0;
        for (int j = 0; j < m; ++j) {
            ++total_used;
            used = j + 1;
            std::vector<double> vj(n);
            std::copy(V.begin() + static_cast<size_t>(j) * n, V.begin() + static_cast<size_t>(j + 1) * n, vj.begin());
            A_work.matvec(vj, av, ug);
            if (!preconditioner.apply(av, w, ug)) return total_used;

            for (int k = 0; k <= j; ++k) {
                const double* vk = V.data() + static_cast<size_t>(k) * n;
                const double h = quantize(dot_quantized_ptr(w, vk, n, ug), ug);
                H[static_cast<size_t>(k) * m + j] = h;
                for (size_t i = 0; i < n; ++i) w[i] = quantize(w[i] - quantize(h * vk[i], ug), ug);
            }
            const double hnext = quantize(norm2(w), ug);
            H[static_cast<size_t>(j + 1) * m + j] = hnext;
            if (hnext > 0.0 && std::isfinite(hnext)) {
                for (size_t i = 0; i < n; ++i) V[static_cast<size_t>(j + 1) * n + i] = quantize(w[i] / hnext, ug);
            }

            for (int k = 0; k < j; ++k) {
                const double h0 = H[static_cast<size_t>(k) * m + j];
                const double h1 = H[static_cast<size_t>(k + 1) * m + j];
                H[static_cast<size_t>(k) * m + j] = quantize(cs[k] * h0 + sn[k] * h1, ug);
                H[static_cast<size_t>(k + 1) * m + j] = quantize(-sn[k] * h0 + cs[k] * h1, ug);
            }
            givens(H[static_cast<size_t>(j) * m + j], H[static_cast<size_t>(j + 1) * m + j], cs[j], sn[j]);
            const double sj = svec[j];
            svec[j] = quantize(cs[j] * sj, ug);
            svec[j + 1] = quantize(-sn[j] * sj, ug);
            H[static_cast<size_t>(j) * m + j] = quantize(cs[j] * H[static_cast<size_t>(j) * m + j] + sn[j] * H[static_cast<size_t>(j + 1) * m + j], ug);
            H[static_cast<size_t>(j + 1) * m + j] = 0.0;

            if (std::abs(svec[j + 1]) / bnrm2 <= gmres_tol || !(hnext > 0.0)) break;
        }

        if (!upper_hessenberg_solve(H, m, svec, used, y)) return total_used;
        for (int k = 0; k < used; ++k) {
            const double* vk = V.data() + static_cast<size_t>(k) * n;
            for (size_t i = 0; i < n; ++i) dx[i] = quantize(dx[i] + quantize(y[k] * vk[i], ug), ug);
        }
    }
    return total_used;
}

static double action_memory_bytes(const Problem& p, const Options& options) {
    const double n = static_cast<double>(p.n);
    const int requested_restart = options.gmres_max_iters <= 0 ? static_cast<int>(p.n) : options.gmres_max_iters;
    const double m = static_cast<double>(std::max(1, std::min(requested_restart, static_cast<int>(p.n))));
    double matrix = storage_bytes(options.u) * n * n;
    if (options.ur != options.u) matrix += storage_bytes(options.ur) * n * n;
    double preconditioner = storage_bytes(options.uf) * n * n + sizeof(int) * n;
    double gmres_workspace = storage_bytes(options.ug) * (n * (m + 1.0) + (m + 1.0) * m) * std::max(1, options.gmres_cycles);
    double vectors = (6.0 * storage_bytes(options.ug) + 3.0 * storage_bytes(options.u) + 2.0 * storage_bytes(options.ur)) * n;
    return matrix + preconditioner + gmres_workspace + vectors;
}

static Row run_case(const Problem& p, const Options& options) {
    Row row;
    row.file = p.file;
    row.split = p.split_id == 0 ? "train" : "test";
    row.n = p.n;
    row.target_cond = p.target_cond;
    row.uf = options.uf;
    row.ug = options.ug;
    row.u = options.u;
    row.ur = options.ur;
    row.memory_bytes = action_memory_bytes(p, options);

    const double anorm = matrix_inf_norm(p.A, p.n);
    const double bnorm = inf_norm(p.b);
    row.a_norm_inf = anorm;
    PrecisionMatrix A_work(p, options.u);
    PrecisionMatrix A_res(p, options.ur);

    std::vector<double> final_x;
    double elapsed_ms = 0.0;
    for (int repeat = -1; repeat < options.repeats; ++repeat) {
        int outer = 0;
        int gmres_iterations = 0;
        std::string solve_status = "ok";
        auto start = Clock::now();
        std::unique_ptr<LUPreconditioner> preconditioner;
        try {
            preconditioner = std::make_unique<LUPreconditioner>(p, options.uf);
        } catch (const std::exception&) {
            solve_status = "factor_fail";
        }
        std::vector<double> x(p.n, 0.0);
        if (preconditioner && !preconditioner->apply(p.b, x, options.u)) {
            solve_status = "solve_fail";
        }
        quantize_vector(x, options.u);
        for (; solve_status == "ok" && outer < options.outer_max_iters; ++outer) {
            std::vector<double> r;
            residual(p, A_res, x, options.ur, r);
            const double ferr = rel_error(x, p.x_true);
            if (std::isfinite(ferr) && ferr <= options.tolerance) {
                break;
            }
            std::vector<double> dx;
            const int inner = gmres_step(A_work, *preconditioner, r, options.ug, options.gmres_max_iters, options.gmres_cycles, options.gmres_tolerance, dx);
            if (inner == 0) {
                solve_status = "inner_fail";
                break;
            }
            gmres_iterations += inner;
            quantize_vector(dx, options.u);
            for (size_t i = 0; i < p.n; ++i) x[i] = quantize(x[i] + dx[i], options.u);
        }
        auto stop = Clock::now();
        if (repeat >= 0) elapsed_ms += std::chrono::duration<double, std::milli>(stop - start).count();
        final_x = std::move(x);
        row.outer_iters = outer;
        row.gmres_iterations = gmres_iterations;
        row.status = solve_status;
    }
    row.avg_ms = elapsed_ms / std::max(1, options.repeats);
    row.rel_error = rel_error(final_x, p.x_true);
    row.backward_error = backward_error(p, final_x, anorm, bnorm);
    if (row.status == "ok") {
        const bool accepted = std::isfinite(row.rel_error) && std::isfinite(row.backward_error)
            && row.rel_error <= options.tolerance;
        if (!accepted) row.status = "not_converged";
    }
    return row;
}

static std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir == ".") return file;
    const char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + file : dir + "/" + file;
}

static void write_rows(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write output: " + path);
    out << "file,split,n,target_cond,uf,ug,u,ur,status,outer_iters,gmres_iterations,avg_ms,rel_error,backward_error,a_norm_inf,memory_bytes\n";
    out << std::setprecision(12);
    for (const auto& row : rows) {
        out << row.file << ',' << row.split << ',' << row.n << ',' << row.target_cond << ','
            << row.uf << ',' << row.ug << ',' << row.u << ',' << row.ur << ',' << row.status << ','
            << row.outer_iters << ',' << row.gmres_iterations << ',' << row.avg_ms << ',' << row.rel_error << ',' << row.backward_error << ','
            << row.a_norm_inf << ',' << row.memory_bytes << '\n';
    }
}

static Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (key == "--manifest") options.manifest = need_value(key);
        else if (key == "--data-dir") options.data_dir = need_value(key);
        else if (key == "--output") options.output = need_value(key);
        else if (key == "--split") options.split = need_value(key);
        else if (key == "--uf") options.uf = need_value(key);
        else if (key == "--ug") options.ug = need_value(key);
        else if (key == "--u") options.u = need_value(key);
        else if (key == "--ur") options.ur = need_value(key);
        else if (key == "--precision") {
            options.uf = options.ug = options.u = options.ur = need_value(key);
        } else if (key == "--repeats") options.repeats = std::stoi(need_value(key));
        else if (key == "--outer-max-iters") options.outer_max_iters = std::stoi(need_value(key));
        else if (key == "--gmres-max-iters") options.gmres_max_iters = std::stoi(need_value(key));
        else if (key == "--gmres-cycles") options.gmres_cycles = std::stoi(need_value(key));
        else if (key == "--gmres-tol" || key == "--gmres-tolerance") options.gmres_tolerance = std::stod(need_value(key));
        else if (key == "--tolerance") options.tolerance = std::stod(need_value(key));
        else if (key == "--backward-tol" || key == "--backward-tolerance") options.backward_tolerance = std::stod(need_value(key));
        else if (key == "--help") {
            std::cout << "Usage: gmres_ir_cpu --manifest manifest.csv --data-dir dir --output out.csv "
                         "[--split train|test] [--uf fp16|fp32|fp64] [--ug ...] [--u ...] [--ur ...] "
                         "[--repeats 3] [--outer-max-iters 10] [--gmres-max-iters 30|0(full)] [--gmres-cycles 1] [--gmres-tol 1e-4] "
                         "[--tolerance 1e-6] [--backward-tol 1e-14 diagnostic-only]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (options.manifest.empty() || options.data_dir.empty() || options.output.empty()) {
        throw std::runtime_error("--manifest, --data-dir, and --output are required");
    }
    if (!valid_precision(options.uf) || !valid_precision(options.ug) || !valid_precision(options.u) || !valid_precision(options.ur)) {
        throw std::runtime_error("uf/ug/u/ur must each be fp16, fp32, or fp64");
    }
    if (options.repeats <= 0 || options.outer_max_iters <= 0 || options.gmres_max_iters < 0 || options.gmres_cycles <= 0) {
        throw std::runtime_error("repeats, outer-max-iters, and gmres-cycles must be positive; gmres-max-iters must be non-negative, with 0 meaning full GMRES");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        std::vector<Row> rows;
        for (const auto& entry : read_manifest(options.manifest)) {
            auto split_it = entry.find("split");
            auto file_it = entry.find("file");
            if (split_it == entry.end() || file_it == entry.end()) continue;
            if (split_it->second != options.split) continue;
            Problem problem;
            problem.file = file_it->second;
            if (!read_problem(join_path(options.data_dir, problem.file), problem)) {
                throw std::runtime_error("failed to read case: " + problem.file);
            }
            rows.push_back(run_case(problem, options));
        }
        write_rows(options.output, rows);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "gmres_ir_cpu: " << ex.what() << '\n';
        return 2;
    }
}
