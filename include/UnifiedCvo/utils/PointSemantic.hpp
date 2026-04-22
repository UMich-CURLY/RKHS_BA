#pragma once

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/impl/point_types.hpp>
#include <Eigen/Core>

namespace pcl {

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
  struct
#ifdef __CUDACC__
  __align__(16)
#else
    alignas(16)
#endif  
  PointSemantic
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PCL_ADD_POINT4D;
    PCL_ADD_RGB;
    float features[FEATURE_DIM];
    int   label;
    float label_distribution[NUM_CLASS];
    float geometric_type[2];
    float normal[3];
    float covariance[9];
    float cov_eigenvalues[3];

    static const unsigned int FEATURE_DIMENSION = FEATURE_DIM;
    static const unsigned int LABEL_DIMENSION   = NUM_CLASS;
    static const unsigned int GEOMETRIC_TYPE_DIMENSION = 2;

    unsigned int feature_dimension()      const { return FEATURE_DIM; }
    unsigned int label_dimension()        const { return NUM_CLASS; }
    unsigned int geometric_type_dimension() const { return 2; }

    // Constructors
#ifdef __CUDACC__
    inline __host__ __device__ PointSemantic() {
#else
    inline PointSemantic() {
#endif
      this->x = this->y = this->z = 0.0f;
      label = -1;
      this->r = this->g = this->b = 0;
      memset(features,           0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(geometric_type,     0, sizeof(float) * 2);
      memset(normal,             0, sizeof(float) * 3);
      memset(covariance,         0, sizeof(float) * 9);
      memset(cov_eigenvalues,    0, sizeof(float) * 3);
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    PointSemantic(float a, float b, float c) {
      this->x = a; this->y = b; this->z = c;
      label = -1;
      this->r = this->g = this->b = 0;
      memset(features,           0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(geometric_type,     0, sizeof(float) * 2);
      memset(normal,             0, sizeof(float) * 3);
      memset(covariance,         0, sizeof(float) * 9);
      memset(cov_eigenvalues,    0, sizeof(float) * 3);
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    PointSemantic(const PointSemantic<FEATURE_DIM, NUM_CLASS> & other) {
      this->x = other.x; this->y = other.y; this->z = other.z;
      label = other.label;
      this->r = other.r; this->g = other.g; this->b = other.b;
      memcpy(features,           other.features,           sizeof(float) * FEATURE_DIM);
      memcpy(label_distribution, other.label_distribution, sizeof(float) * NUM_CLASS);
      memcpy(geometric_type,     other.geometric_type,     sizeof(float) * 2);
      memcpy(normal,             other.normal,             sizeof(float) * 3);
      memcpy(covariance,         other.covariance,         sizeof(float) * 9);
      memcpy(cov_eigenvalues,    other.cov_eigenvalues,    sizeof(float) * 3);
    }
  };

  // -------------------------------------------------------------------------
  // Conversion helpers (renamed accordingly)
  // -------------------------------------------------------------------------
  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZRGB>
  void PointSemantic_to_PointXYZRGB(const pcl::PointCloud<pcl::PointSemantic<FEATURE_DIM, NUM_CLASS>> & pc_seg,
                                    typename pcl::PointCloud<PointWithXYZRGB> & pc_rgb) {
    pc_rgb.resize(pc_seg.size());
    for (size_t i = 0; i < pc_seg.size(); ++i) {
      auto & p_rgb = pc_rgb[i];
      auto & p_seg = pc_seg[i];
      p_rgb.x = p_seg.x; p_rgb.y = p_seg.y; p_rgb.z = p_seg.z;
      p_rgb.r = p_seg.r; p_rgb.g = p_seg.g; p_rgb.b = p_seg.b;
    }
    pc_rgb.header = pc_seg.header;
  }

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZI>
  void PointSemantic_to_PointXYZI(const pcl::PointCloud<pcl::PointSemantic<FEATURE_DIM, NUM_CLASS>> & pc_seg,
                                  typename pcl::PointCloud<PointWithXYZI> & pc_i) {
    pc_i.resize(pc_seg.size());
    for (size_t i = 0; i < pc_seg.size(); ++i) {
      auto & p_i   = pc_i[i];
      auto & p_seg = pc_seg[i];
      p_i.x = p_seg.x; p_i.y = p_seg.y; p_i.z = p_seg.z;
      p_i.intensity = static_cast<uint8_t>(p_seg.features[0] * 255.0f);
    }
    pc_i.header = pc_seg.header;
  }

  // ... (other conversion functions similarly renamed)

} // namespace pcl
