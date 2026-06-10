// npaged-core/kernel.cpp
// Batched cosine-similarity kernels for normalized embeddings (cosine == dot).
//
//   cosine_similarity_scalar   : plain C++ baseline (reference)
//   cosine_similarity_neon_st  : single-threaded NEON+FMA
//   cosine_similarity_neon     : multi-threaded NEON+FMA (n_threads arg)
//
// All take a normalized query (dim,) and candidates (n, dim) row-major,
// return (n,) scores. Multi-threaded version partitions candidate ROWS across
// threads; each thread writes a disjoint output slice, so no locks are needed.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

namespace py = pybind11;

// One candidate's dot product with the query, NEON + FMA. Pure C++ helper.
static inline float dot_neon(const float* q, const float* row, ssize_t dim) {
#if HAVE_NEON
    const ssize_t vec_end = (dim / 4) * 4;
    float32x4_t acc4 = vdupq_n_f32(0.0f);
    ssize_t d = 0;
    for (; d < vec_end; d += 4) {
        float32x4_t qv = vld1q_f32(q + d);
        float32x4_t cv = vld1q_f32(row + d);
        acc4 = vfmaq_f32(acc4, qv, cv);   // fused multiply-add: acc4 += qv*cv
    }
    float acc = vaddvq_f32(acc4);
    for (; d < dim; ++d) acc += q[d] * row[d];  // tail
    return acc;
#else
    float acc = 0.0f;
    for (ssize_t d = 0; d < dim; ++d) acc += q[d] * row[d];
    return acc;
#endif
}

// Compute scores for candidate rows [start, end) into out[start..end).
static void worker(const float* q, const float* c, float* out,
                   ssize_t start, ssize_t end, ssize_t dim) {
    for (ssize_t i = start; i < end; ++i) {
        out[i] = dot_neon(q, c + i * dim, dim);
    }
}

// ---- Scalar baseline -------------------------------------------------------
py::array_t<float> cosine_similarity_scalar(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> candidates) {
    py::buffer_info q_info = query.request(), c_info = candidates.request();
    if (q_info.ndim != 1) throw std::runtime_error("query must be 1-D");
    if (c_info.ndim != 2) throw std::runtime_error("candidates must be 2-D");
    const ssize_t dim = q_info.shape[0], n = c_info.shape[0];
    if (c_info.shape[1] != dim) throw std::runtime_error("dim mismatch");
    const float* q = static_cast<const float*>(q_info.ptr);
    const float* c = static_cast<const float*>(c_info.ptr);
    auto result = py::array_t<float>(n);
    float* out = static_cast<float*>(result.request().ptr);
    for (ssize_t i = 0; i < n; ++i) {
        const float* row = c + i * dim;
        float acc = 0.0f;
        for (ssize_t d = 0; d < dim; ++d) acc += q[d] * row[d];
        out[i] = acc;
    }
    return result;
}

// ---- Single-threaded NEON+FMA ----------------------------------------------
py::array_t<float> cosine_similarity_neon_st(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> candidates) {
    py::buffer_info q_info = query.request(), c_info = candidates.request();
    if (q_info.ndim != 1) throw std::runtime_error("query must be 1-D");
    if (c_info.ndim != 2) throw std::runtime_error("candidates must be 2-D");
    const ssize_t dim = q_info.shape[0], n = c_info.shape[0];
    if (c_info.shape[1] != dim) throw std::runtime_error("dim mismatch");
    const float* q = static_cast<const float*>(q_info.ptr);
    const float* c = static_cast<const float*>(c_info.ptr);
    auto result = py::array_t<float>(n);
    float* out = static_cast<float*>(result.request().ptr);
    worker(q, c, out, 0, n, dim);
    return result;
}

// ---- Multi-threaded NEON+FMA -----------------------------------------------
// n_threads=0 -> auto (hardware_concurrency). Falls back to single-threaded
// below a size threshold, where thread-launch overhead would dominate.
py::array_t<float> cosine_similarity_neon(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> candidates,
    int n_threads = 0) {
    py::buffer_info q_info = query.request(), c_info = candidates.request();
    if (q_info.ndim != 1) throw std::runtime_error("query must be 1-D");
    if (c_info.ndim != 2) throw std::runtime_error("candidates must be 2-D");
    const ssize_t dim = q_info.shape[0], n = c_info.shape[0];
    if (c_info.shape[1] != dim) throw std::runtime_error("dim mismatch");
    const float* q = static_cast<const float*>(q_info.ptr);
    const float* c = static_cast<const float*>(c_info.ptr);
    auto result = py::array_t<float>(n);
    float* out = static_cast<float*>(result.request().ptr);

    // Decide thread count.
    int T = n_threads;
    if (T <= 0) {
        // Measured on M3 Pro: speedup peaks at the P-core count (5), and
        // degrades past it as efficiency-core chunks become stragglers.
        // hardware_concurrency() returns ALL cores (11), which is slower.
        T = 5;
    }
    const ssize_t PARALLEL_THRESHOLD = 8192;  // measured: threading only pays above ~10K candidates
    if (n < PARALLEL_THRESHOLD || T <= 1) {
        worker(q, c, out, 0, n, dim);
        return result;
    }

    // Release the GIL so threads run truly in parallel (no Python objects
    // are touched inside the workers — only raw float buffers).
    {
        py::gil_scoped_release release;
        std::vector<std::thread> threads;
        threads.reserve(T);
        const ssize_t chunk = (n + T - 1) / T;  // ceil division
        for (int t = 0; t < T; ++t) {
            ssize_t start = static_cast<ssize_t>(t) * chunk;
            ssize_t end = std::min(start + chunk, n);
            if (start >= end) break;
            threads.emplace_back(worker, q, c, out, start, end, dim);
        }
        for (auto& th : threads) th.join();
    }
    return result;
}

PYBIND11_MODULE(npaged_core, m) {
    m.doc() = "Neuro-Paging native core (NEON-accelerated distance kernels)";
    m.def("cosine_similarity_scalar", &cosine_similarity_scalar,
          "Batched cosine similarity (scalar baseline)");
    m.def("cosine_similarity_neon_st", &cosine_similarity_neon_st,
          "Batched cosine similarity (single-threaded NEON+FMA)");
    m.def("cosine_similarity_neon", &cosine_similarity_neon,
          py::arg("query"), py::arg("candidates"), py::arg("n_threads") = 0,
          "Batched cosine similarity (multi-threaded NEON+FMA)");
    m.attr("has_neon") = bool(HAVE_NEON);
}
