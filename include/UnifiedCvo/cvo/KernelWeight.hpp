#pragma once

#include <cmath>

#include "cvo/CvoParams.hpp"

namespace cvo {
namespace detail {

template <typename PointT>
inline __host__ __device__ float point_feature_distance2(const PointT& a, const PointT& b) {
    float d2 = 0.0f;
    for (unsigned int i = 0; i < PointT::FEATURE_DIMENSION; ++i) {
        const float diff = a.features[i] - b.features[i];
        d2 += diff * diff;
    }
    return d2;
}

template <typename PointT>
inline __host__ __device__ float point_semantic_distance2(const PointT& a, const PointT& b) {
    float d2 = 0.0f;
    for (unsigned int i = 0; i < PointT::LABEL_DIMENSION; ++i) {
        const float diff = a.label_distribution[i] - b.label_distribution[i];
        d2 += diff * diff;
    }
    return d2;
}

template <typename PointT>
inline __host__ __device__ float point_geometric_type_ip(const PointT& a, const PointT& b) {
    float ip = 0.0f;
    for (unsigned int i = 0; i < PointT::GEOMETRIC_TYPE_DIMENSION; ++i) {
        ip += a.geometric_type[i] * b.geometric_type[i];
    }
    return ip;
}

inline __host__ __device__ float compute_range_ell_value(float ell,
                                                         float x,
                                                         float y,
                                                         float z,
                                                         float min_r,
                                                         float max_r) {
    const float r = sqrtf(x * x + y * y + z * z);
    const float clamped_r = fminf(fmaxf(r, min_r), max_r);
    return ell * (clamped_r / 5.0f + 1.0f);
}

template <typename PointT>
inline __host__ __device__ float combined_kernel_weight(const CvoParams& params,
                                                        const PointT& pa,
                                                        const PointT& pb,
                                                        float ell) {
    const float sp_thres = params.sp_thres;
    float weight = 1.0f;

    if (params.is_using_geometric_type) {
        const float geo_type = point_geometric_type_ip(pa, pb);
        if (geo_type < 0.001f) {
            return 0.0f;
        }
        weight *= geo_type;
    }

    if (params.is_using_geometry) {
        const float sigma2 = params.sigma * params.sigma;
        const float spatial_ell = params.is_using_range_ell
            ? compute_range_ell_value(ell, pa.x, pa.y, pa.z, 1.0f, 80.0f)
            : ell;
        const float d2_thres = -2.0f * spatial_ell * spatial_ell * logf(sp_thres / sigma2);
        const float dx = pa.x - pb.x;
        const float dy = pa.y - pb.y;
        const float dz = pa.z - pb.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= d2_thres) {
            return 0.0f;
        }
        weight *= sigma2 * expf(-d2 / (2.0f * spatial_ell * spatial_ell));
    }

    if (params.is_using_intensity) {
        const float sigma2 = params.c_sigma * params.c_sigma;
        const float ell2 = params.c_ell * params.c_ell;
        const float d2_thres = -2.0f * ell2 * logf(sp_thres / sigma2);
        const float d2 = point_feature_distance2(pa, pb);
        if (d2 >= d2_thres) {
            return 0.0f;
        }
        weight *= sigma2 * expf(-d2 / (2.0f * ell2));
    }

    if (params.is_using_semantics) {
        const float sigma2 = params.s_sigma * params.s_sigma;
        const float ell2 = params.s_ell * params.s_ell;
        const float d2_thres = -2.0f * ell2 * logf(sp_thres / sigma2);
        const float d2 = point_semantic_distance2(pa, pb);
        if (d2 >= d2_thres) {
            return 0.0f;
        }
        weight *= sigma2 * expf(-d2 / (2.0f * ell2));
    }

    return (weight > sp_thres) ? weight : 0.0f;
}

}  // namespace detail
}  // namespace cvo
