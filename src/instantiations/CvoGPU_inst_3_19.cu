// src/instantiations/CvoGPU_inst_3_19.cu

// No global macros. Define constants locally.
namespace {
    constexpr unsigned int FEATURE_DIM = 3;
    constexpr unsigned int NUM_CLASS = 19;
}

#include "utils/PointSemantic.hpp"
#include "cvo/CvoGPU.cuh"

// Define point type local to this translation unit
using CvoPoint_3_19 = pcl::PointSemantic<FEATURE_DIM, NUM_CLASS>;

// Register with PCL (must be in global namespace)
POINT_CLOUD_REGISTER_POINT_STRUCT(CvoPoint_3_19,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, rgb, rgb)
    (float[FEATURE_DIM], features, features)
    (int, label, label)
    (float[NUM_CLASS], label_distribution, label_distribution)
    (float[2], geometric_type, geometric_type)
    (float[3], normal, normal)
    (float[9], covariance, covariance)
    (float[3], cov_eigenvalues, cov_eigenvalues)
)

// Explicit instantiation
template class cvo::CvoGPU<CvoPoint_3_19>;
