#pragma once

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
//#include <boost/shared_ptr.hpp>
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

  PointSegmentedDistribution
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned    
    // data
    PCL_ADD_POINT4D;                 
    PCL_ADD_RGB;
    float features[FEATURE_DIM];
    int   label;
    float label_distribution[NUM_CLASS];
    float geometric_type[2]; // edge: 0; surface: 1
    float normal[3];
    float covariance[9];
    float cov_eigenvalues[3];

    static const unsigned int FEATURE_DIMENSION = FEATURE_DIM;
    static const unsigned int LABEL_DIMENSION = NUM_CLASS;
    static const unsigned int GEOMETRIC_TYPE_DIMENSION = 2;

    unsigned int feature_dimension() const {return  FEATURE_DIM;}
    unsigned int label_dimension() const { return  NUM_CLASS;}
    unsigned int geometric_type_dimension() const {return  2;}


      


    // methods
#ifdef __CUDACC__ 
    inline __host__ __device__ PointSegmentedDistribution() {
#else
    inline PointSegmentedDistribution() {
#endif
      this->x = 0;
      this->y = 0;
      this->z = 0;
      label = -1;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      memset(features, 0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(geometric_type, 0, sizeof(float)*2);
      memset(normal, 0, sizeof(float)*3);
      memset(covariance, 0, sizeof(float)*9);
      memset(cov_eigenvalues, 0, sizeof(float)*3);
    }

#ifdef __CUDACC__
    __host__ __device__ 
#endif
    PointSegmentedDistribution(float a, float b, float c) {
      this->x = a;
      this->y = b;
      this->z = c;
      label = -1;
      this->r = 0;
      this->g = 0;
      this->b = 0;
      memset(features, 0, sizeof(float) * FEATURE_DIM);
      memset(label_distribution, 0, sizeof(float) * NUM_CLASS);
      memset(normal, 0, sizeof(float)*3);
      memset(covariance, 0, sizeof(float)*9);
      memset(cov_eigenvalues, 0, sizeof(float)*3);
    }
    
#ifdef __CUDACC__
    __host__ __device__ 
#endif
      PointSegmentedDistribution(const PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS> & other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
      label = other.label;
      this->r = other.r;
      this->g = other.g;
      this->b = other.b;
      memcpy(features, other.features, sizeof(float) * FEATURE_DIM);
      memcpy(label_distribution, other.label_distribution, sizeof(float) * NUM_CLASS);
      memcpy(geometric_type, other.geometric_type, sizeof(float)*2);      
      memcpy(normal, other.normal, sizeof(float)*3);
      memcpy(covariance, other.covariance, sizeof(float)*9);
      memcpy(cov_eigenvalues, other.cov_eigenvalues, sizeof(float)*3);
    }

    
    };
    //EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
    

    
  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZRGB>
  void PointSeg_to_PointXYZRGB(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>> & pc_seg,
                               typename pcl::PointCloud<PointWithXYZRGB> & pc_rgb) {
    pc_rgb.resize(pc_seg.size());
    for (int i = 0; i < pc_rgb.size(); i++) {
      auto & p_rgb = pc_rgb[i];
      auto & p_seg = pc_seg[i];

      p_rgb.x = p_seg.x;
      p_rgb.y = p_seg.y;
      p_rgb.z = p_seg.z;
      p_rgb.r = p_seg.r;
      p_rgb.g = p_seg.g;
      p_rgb.b = p_seg.b;
      
    }
    pc_rgb.header = pc_seg.header;
  }

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZI>
  void PointSeg_to_PointXYZI(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>> & pc_seg,
                               typename pcl::PointCloud<PointWithXYZI> & pc_i) {
    pc_i.resize(pc_seg.size());
    for (int i = 0; i < pc_i.size(); i++) {
      auto & p_i = pc_i[i];
      auto & p_seg = pc_seg[i];
      p_i.x = p_seg.x;
      p_i.y = p_seg.y;
      p_i.z = p_seg.z;
      p_i.intensity = p_seg.features[0] * 255.0;
    }
    pc_i.header = pc_seg.header;
  }

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZI>
  void PointSeg_from_PointXYZI(
                               const typename pcl::PointCloud<PointWithXYZI> & pc_i,
                               pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>> & pc_seg
                               ) {
    pc_seg.resize(pc_i.size());
    for (int i = 0; i < pc_i.size(); i++) {
      auto & p_seg = pc_seg[i];
      auto & p_i = pc_i[i];
      p_seg.x = p_i.x;
      p_seg.y = p_i.y;
      p_seg.z = p_i.z;
      p_seg.features[0] = p_i.intensity / 255.0;
    }
    pc_seg.header = pc_i.header;
  }
  
  

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS, typename PointWithXYZRGB>
    void PointSeg_from_PointXYZRGB(const typename pcl::PointCloud<PointWithXYZRGB> & pc_rgb,
                                   pcl::PointCloud<pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>> & pc_seg
                                 ) {
    pc_seg.resize(pc_rgb.size());
    for (int i = 0; i < pc_seg.size(); i++) {
      auto & p_rgb = pc_rgb[i];
      auto & p_seg = pc_seg[i];

      p_seg.x = p_rgb.x;
      p_seg.y = p_rgb.y;
      p_seg.z = p_rgb.z;
      p_seg.rgb = p_rgb.rgb;
      p_seg.features[0] = ((float)(int)p_rgb.r) / 255.0;
      p_seg.features[1] = ((float)(int)p_rgb.g) / 255.0;
      p_seg.features[2] = ((float)(int)p_rgb.b) / 255.0;
      p_seg.geometric_type[0] = 0;
      p_seg.geometric_type[1] = 1;
      
    }
    pc_seg.header = pc_rgb.header;
  }
  

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
    unsigned int feature_dimension(const pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>& p) {
    return FEATURE_DIM;
  }

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
    unsigned int label_dimension(const pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>& p) {
    return NUM_CLASS;
  }

  template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
    unsigned int geometric_type_dimension(const pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>& p) {
    return 2;
  }
  

  template <unsigned int FEATRURE_DIM, unsigned int NUM_CLASS >
  void PointSeg_to_PointXYZ(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATRURE_DIM, NUM_CLASS>> & pc_seg,
                            pcl::PointCloud<pcl::PointXYZ> & pc) {
    pc.resize(pc_seg.size());
    for (int i = 0; i < pc.size(); i++) {
      auto & p = pc[i];
      auto & p_seg = pc_seg[i];

      p.x = p_seg.x;
      p.y = p_seg.y;
      p.z = p_seg.z;
    }
    pc.header = pc_seg.header;
  }
    
    template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS >
    void print_point(const pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS> & p) {
      std::cout<<"The point is at ("<<p.x<<", "<<p.y<<", "<<p.z<<")\n";
      std::cout<<"the features are ";
      for (int i = 0; i < FEATURE_DIM; i++)
        std::cout<<p.features[i]<<", ";
      std::cout<<std::endl;
      std::cout<<"the semantic distribution is ";
      for (int i = 0; i < NUM_CLASS; i++)
        std::cout<<p.label_distribution[i]<<", ";
      std::cout<<std::endl;
      std::cout<<"geometric type is ";
      for (int i = 0; i < 2; i++)
        std::cout<<p.geometric_type[i]<<" ";

      std::cout<<"\ncovariance matrix is \n";
      for (int i = 0; i < 9; i++) {
        if (i && i%3==0) std::cout<<std::endl;
        std::cout<<p.covariance[i]<<"  ";
      }
      std::cout<<"eigen values of the covariance matrix are ";
      for (int i = 0; i < 3; i++) {
        std::cout<<p.cov_eigenvalues[i]<<"  ";
      }

      std::cout<<std::endl;
                           
    }


    template <unsigned int FEATRURE_DIM, unsigned int NUM_CLASS>
      void PointSeg_to_PCL(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATRURE_DIM, NUM_CLASS>> & pc_seg,
                                           pcl::PointCloud<pcl::PointXYZI> & pc_i) {
      pc_i.resize(pc_seg.size());
      for (int i = 0; i < pc_i.size(); i++) {
        auto & p_i = pc_i[i];
        auto & p_seg = pc_seg[i];
        p_i.x = p_seg.x;
        p_i.y = p_seg.y;
        p_i.z = p_seg.z;
        p_i.intensity = p_seg.features[0] * 255.0;
      }
      pc_i.header = pc_seg.header;
    }

    template <unsigned int FEATRURE_DIM, unsigned int NUM_CLASS>
      void PointSeg_to_PCL(const pcl::PointCloud<pcl::PointSegmentedDistribution<FEATRURE_DIM, NUM_CLASS>> & pc_seg,
                           pcl::PointCloud<pcl::PointXYZRGB> & pc_rgb) {
      pc_rgb.resize(pc_seg.size());
      for (int i = 0; i < pc_rgb.size(); i++) {
        auto & p_rgb = pc_rgb[i];
        auto & p_seg = pc_seg[i];

        p_rgb.x = p_seg.x;
        p_rgb.y = p_seg.y;
        p_rgb.z = p_seg.z;
        p_rgb.r = p_seg.r;
        p_rgb.g = p_seg.g;
        p_rgb.b = p_seg.b;
      
      }
      pc_rgb.header = pc_seg.header;
    }
    


    

}


    
