// include/UnifiedCvo/cvo/CvoState.cuh
#pragma once

#include <thrust/device_vector.h>
#include <Eigen/Dense>
#include <memory>

#include "cvo/CvoParams.hpp"
#include "cvo/SparseKernelMat.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cukdtree/cukdtree.cuh"   // assuming available

namespace cvo {

template <typename PointT>
struct CvoState {
    using PointCloud = CvoPointCloud<PointT>;

    // Host parameters
    double dl;
    float step;
    int num_fixed;
    int num_moving;
    float ell;
    float ell_max;
    bool is_ell_adaptive;

    // GPU raw pointers
    SparseKernelMat *A;
    SparseKernelMat *Axx;   // only if adaptive
    SparseKernelMat *Ayy;
    SparseKernelMat A_host, Axx_host, Ayy_host;

    Eigen::Matrix3f *R_gpu;
    Eigen::Vector3f *T_gpu;
    Eigen::Vector3f *omega;
    Eigen::Vector3f *v;

    // Thrust device vectors
    thrust::device_vector<PointT> cloud_x_gpu;
    thrust::device_vector<PointT> cloud_y_gpu;
    thrust::device_vector<PointT> cloud_y_gpu_init;
    std::shared_ptr<perl_registration::cuKdTree<PointT>> kdtree_moving_points;
    thrust::device_vector<PointT> cloud_x_gpu_transformed_kdtree;
    thrust::device_vector<int> kdtree_inds_results;

    thrust::device_vector<Eigen::Vector3d> omega_gpu;
    thrust::device_vector<Eigen::Vector3d> v_gpu;

    // Temporary buffers for step size computation
    thrust::device_vector<Eigen::Matrix<float,1,3>> xiz;
    thrust::device_vector<Eigen::Matrix<float,1,3>> xi2z;
    thrust::device_vector<Eigen::Matrix<float,1,3>> xi3z;
    thrust::device_vector<Eigen::Matrix<float,1,3>> xi4z;
    thrust::device_vector<float> normxiz2;
    thrust::device_vector<float> xiz_dot_xi2z;
    thrust::device_vector<float> epsil_const;
    thrust::device_vector<double> B, C, D, E;

    CvoState(std::shared_ptr<PointCloud> source,
             std::shared_ptr<PointCloud> target,
             const CvoParams &params);
    ~CvoState();

    void reset_state_at_new_iter(int num_neighbors);
};

// Implementation follows in the same file (header-only)
template <typename PointT>
CvoState<PointT>::CvoState(std::shared_ptr<PointCloud> source,
                           std::shared_ptr<PointCloud> target,
                           const CvoParams &params)
    : dl(params.dl), step(0),
      num_fixed(source->size()), num_moving(target->size()),
      ell(params.ell_init), ell_max(params.ell_max),
      is_ell_adaptive(params.is_ell_adaptive),
      omega_gpu(num_fixed, Eigen::Vector3d::Zero()),
      v_gpu(num_fixed, Eigen::Vector3d::Zero()),
      xiz(num_moving, Eigen::Matrix<float,1,3>::Zero()),
      xi2z(num_moving, Eigen::Matrix<float,1,3>::Zero()),
      xi3z(num_moving, Eigen::Matrix<float,1,3>::Zero()),
      xi4z(num_moving, Eigen::Matrix<float,1,3>::Zero()),
      normxiz2(num_moving, 0),
      xiz_dot_xi2z(num_moving, 0),
      epsil_const(num_moving, 0),
      B(num_fixed, 0), C(num_fixed, 0), D(num_fixed, 0), E(num_fixed, 0)
{
    int A_rows = num_fixed;
    int A_cols = params.is_full_ip_matrix ? num_moving : params.nearest_neighbors_max;
    A = init_SparseKernelMat_gpu(A_rows, A_cols, A_host);
    if (is_ell_adaptive) {
        Axx = init_SparseKernelMat_gpu(A_rows, A_cols, Axx_host);
        Ayy = init_SparseKernelMat_gpu(num_moving, A_cols, Ayy_host);
    }
    cudaMalloc(&R_gpu, sizeof(Eigen::Matrix3f));
    cudaMalloc(&T_gpu, sizeof(Eigen::Vector3f));
    cudaMalloc(&omega, sizeof(Eigen::Vector3f));
    cudaMalloc(&v, sizeof(Eigen::Vector3f));

    // Copy point clouds to device
    cloud_x_gpu = source->points();
    cloud_y_gpu_init = target->points();
    cloud_y_gpu.resize(num_moving);
    thrust::copy(cloud_y_gpu_init.begin(), cloud_y_gpu_init.end(), cloud_y_gpu.begin());

    if (params.is_using_kdtree) {
        kdtree_moving_points = std::make_shared<perl_registration::cuKdTree<PointT>>();
        kdtree_moving_points->SetInputCloud(cloud_y_gpu_init);
        cloud_x_gpu_transformed_kdtree.resize(num_fixed);
        kdtree_inds_results.resize(params.is_using_kdtree * num_fixed);
    }
}

template <typename PointT>
CvoState<PointT>::~CvoState() {
    cudaFree(R_gpu);
    cudaFree(T_gpu);
    cudaFree(omega);
    cudaFree(v);
    delete_SparseKernelMat_gpu(A, &A_host);
    if (is_ell_adaptive) {
        delete_SparseKernelMat_gpu(Axx, &Axx_host);
        delete_SparseKernelMat_gpu(Ayy, &Ayy_host);
    }
}

template <typename PointT>
void CvoState<PointT>::reset_state_at_new_iter(int num_neighbors) {
    clear_SparseKernelMat(&A_host, num_neighbors);
    if (is_ell_adaptive) {
        clear_SparseKernelMat(&Axx_host, num_neighbors);
        clear_SparseKernelMat(&Ayy_host, num_neighbors);
    }
    if (kdtree_moving_points) {
        cudaMemset(thrust::raw_pointer_cast(kdtree_inds_results.data()), -1,
                   sizeof(int) * kdtree_inds_results.size());
    }
}

} // namespace cvo
