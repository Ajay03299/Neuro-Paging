// npaged-core/kernel.cpp
// Batched cosine-similarity kernels. Because the embeddings are L2-normalized,
// cosine similarity == dot product, so each score is sum(query[d]*cand[i][d]).
//
//   cosine_similarity_scalar : plain C++ baseline (correct reference)
//   cosine_similarity_neon   : ARM NEON SIMD, 4 floats per instruction
//
// Both take a normalized query (dim,) and candidate matrix (n, dim) row-major,
// and return (n,) similarity scores. The NEON version must match the scalar
// version (up to float32 rounding).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

namespace py = pybind11;

// ---- Scalar baseline (Phase 3) ---------------------------------------------
py::array_t<float> cosine_similarity_scalar(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> candidates) {

    py::buffer_info q_info = query.request();
    py::buffer_info c_info = candidates.request();
    if (q_info.ndim != 1) throw std::runtime_error("query must be 1-D");
    if (c_info.ndim != 2) throw std::runtime_error("candidates must be 2-D");

    const ssize_t dim = q_info.shape[0];
    const ssize_t n = c_info.shape[0];
    if (c_info.shape[1] != dim)
        throw std::runtime_error("candidate dim must match query dim");

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

// ---- NEON SIMD version (Phase 4) -------------------------------------------
py::array_t<float> cosine_similarity_neon(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> candidates) {

    py::buffer_info q_info = query.request();
    py::buffer_info c_info = candidates.request();
    if (q_info.ndim != 1) throw std::runtime_error("query must be 1-D");
    if (c_info.ndim != 2) throw std::runtime_error("candidates must be 2-D");

    const ssize_t dim = q_info.shape[0];
    const ssize_t n = c_info.shape[0];
    if (c_info.shape[1] != dim)
        throw std::runtime_error("candidate dim must match query dim");

    const float* q = static_cast<const float*>(q_info.ptr);
    const float* c = static_cast<const float*>(c_info.ptr);
    auto result = py::array_t<float>(n);
    float* out = static_cast<float*>(result.request().ptr);

#if HAVE_NEON
    const ssize_t vec_end = (dim / 4) * 4;  // largest multiple of 4 <= dim
    for (ssize_t i = 0; i < n; ++i) {
        const float* row = c + i * dim;
        // acc4 holds 4 running partial sums, one per lane.
        float32x4_t acc4 = vdupq_n_f32(0.0f);  // {0,0,0,0}
        ssize_t d = 0;
        for (; d < vec_end; d += 4) {
            float32x4_t qv = vld1q_f32(q + d);     // load 4 query floats
            float32x4_t cv = vld1q_f32(row + d);   // load 4 candidate floats
            acc4 = vmlaq_f32(acc4, qv, cv);        // acc4 += qv * cv (4 lanes)
        }
        float acc = vaddvq_f32(acc4);  // horizontal sum of the 4 lanes
        for (; d < dim; ++d) acc += q[d] * row[d];  // tail (if dim % 4 != 0)
        out[i] = acc;
    }
#else
    // Fallback if compiled on a non-NEON target.
    for (ssize_t i = 0; i < n; ++i) {
        const float* row = c + i * dim;
        float acc = 0.0f;
        for (ssize_t d = 0; d < dim; ++d) acc += q[d] * row[d];
        out[i] = acc;
    }
#endif
    return result;
}

PYBIND11_MODULE(npaged_core, m) {
    m.doc() = "Neuro-Paging native core (NEON-accelerated distance kernels)";
    m.def("cosine_similarity_scalar", &cosine_similarity_scalar,
          "Batched cosine similarity (scalar baseline)");
    m.def("cosine_similarity_neon", &cosine_similarity_neon,
          "Batched cosine similarity (ARM NEON SIMD)");
    m.attr("has_neon") = bool(HAVE_NEON);
}
