// src/cvo/CvoGPU_instantiations.cu
#include "cvo/CvoGPU.cuh"
#include "utils/PointSemantic.hpp"

// Instantiate for FEATURE_DIM=3, NUM_CLASS=19
template class cvo::CvoGPU<pcl::PointSemantic<3, 19>>;

// Instantiate for FEATURE_DIM=1, NUM_CLASS=19
template class cvo::CvoGPU<pcl::PointSemantic<1, 19>>;
