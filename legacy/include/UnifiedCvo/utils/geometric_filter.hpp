#pragma once
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include <Eigen/Dense>
#include <vector>
#include "cvo/Association.hpp"
//#include <iostream>

namespace cvo {
  void match_two_frame(const cvo::CvoGPU & cvo_align,
                       cvo::CvoPointCloud & pc1,
                       cvo::CvoPointCloud & pc2,
                       float depth_normal_ell,
                       float depth_dir_ell,
                       const Eigen::Matrix4d & T1,
                       const Eigen::Matrix4d & T2,
                       bool is_filling_pc2_inlier,
                       // outputs
                       std::vector<bool> & pc1_inliers,
                       std::vector<bool> & pc2_inliers
                       ) {
    if (pc1_inliers.size() == 0)
      pc1_inliers.resize(pc1.size(), false);

    if (pc2_inliers.size() == 0 && is_filling_pc2_inlier)
      pc2_inliers.resize(pc2.size(), false);
  
    Eigen::Matrix3f non_isotropic_kernel= Eigen::Matrix3f::Identity();
    non_isotropic_kernel(0,0) = depth_normal_ell;
    non_isotropic_kernel(1,1) = depth_normal_ell;
    non_isotropic_kernel(2,2) = depth_dir_ell;    
    //std::cout<<"kernel is "<<non_isotropic_kernel<<std::endl;


    Eigen::Matrix4f T_t2s = (T2.inverse() * T1).cast<float>();
    Eigen::Matrix4f T_s2t = (T1.inverse() * T2).cast<float>();

    cvo::Association association;
    cvo_align.compute_association_gpu(pc1,
                                      pc2,
                                      T_t2s,
                                      non_isotropic_kernel,
                                      association
                                      );
    //std::cout<<" non kf "<<i<<" has nonzeros "<<association.pairs.nonZeros()<<std::endl;

    for (int k=0; k<association.pairs.outerSize(); ++k)
    {
      for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(association.pairs,k); it; ++it) {

        int idx1 = it.row();
        int idx2 = it.col();
        pc1_inliers[idx1] = true;
        if (is_filling_pc2_inlier)
          pc2_inliers[idx2] = true;
      }
    }

  

 
  }

}
