// include/UnifiedCvo/cvo/CvoGPU.cuh
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <limits>
#include <unordered_map>

#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/const_def.hpp"
#include "cvo/Association.hpp"
#include "cvo/AssociationUtils.hpp"
#include "cvo/CvoState.cuh"
#include "cvo/KernelWeight.hpp"
#include "cvo/LieGroup.h"
#include "cvo/SparsePoseAssembly.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/SparseKernelMat.hpp"

namespace cvo {

// Forward declaration of helper functions
template <typename PointT>
void update_tf(const Eigen::Matrix3f &R, const Eigen::Vector3f &T,
               CvoState<PointT> *state, Eigen::Ref<Eigen::Matrix4f> transform);

template <typename PointT>
__global__ void fill_in_A_mat_gpu(const CvoParams *params,
                                  const PointT *points_a, int a_size,
                                  const PointT *points_b, int b_size,
                                  int num_neighbors, float ell,
                                  SparseKernelMat *A_mat);

template <typename PointT>
__global__ void compute_flow_gpu(const CvoParams *params,
                                 const PointT *cloud_x, const PointT *cloud_y,
                                 const SparseKernelMat *A, int num_neighbors,
                                 Eigen::Vector3d *omega_gpu, Eigen::Vector3d *v_gpu);

template <typename PointT>
__global__ void transform_pointcloud_kernel(const PointT *input, PointT *output,
                                            int num_points,
                                            const Eigen::Matrix3f *R, const Eigen::Vector3f *T);


// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------
template <typename PointT>
CvoGPU<PointT>::CvoGPU(const CvoParams &params) : params_(params) {
    cudaMalloc(&params_gpu_, sizeof(CvoParams));
    cudaMemcpy(params_gpu_, &params_, sizeof(CvoParams), cudaMemcpyHostToDevice);
}

template <typename PointT>
CvoGPU<PointT>::~CvoGPU() {
    release_multiframe_system_buffers();
    cudaFree(params_gpu_);
}

template <typename PointT>
void CvoGPU<PointT>::ensure_multiframe_system_buffers(int system_size) {
    if (multiframe_system_size_ == system_size &&
        multiframe_H_device_ != nullptr &&
        multiframe_g_device_ != nullptr &&
        multiframe_cost_device_ != nullptr) {
        return;
    }
    release_multiframe_system_buffers();
    cudaMalloc(reinterpret_cast<void**>(&multiframe_H_device_),
               static_cast<size_t>(system_size) * static_cast<size_t>(system_size) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&multiframe_g_device_),
               static_cast<size_t>(system_size) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&multiframe_cost_device_), sizeof(double));
    multiframe_system_size_ = system_size;
}

template <typename PointT>
void CvoGPU<PointT>::release_multiframe_system_buffers() {
    if (multiframe_H_device_ != nullptr) {
        cudaFree(multiframe_H_device_);
        multiframe_H_device_ = nullptr;
    }
    if (multiframe_g_device_ != nullptr) {
        cudaFree(multiframe_g_device_);
        multiframe_g_device_ = nullptr;
    }
    if (multiframe_cost_device_ != nullptr) {
        cudaFree(multiframe_cost_device_);
        multiframe_cost_device_ = nullptr;
    }
    multiframe_system_size_ = 0;
}

template <typename PointT>
void update_tf(const Eigen::Matrix3f &R,
               const Eigen::Vector3f &T,
               CvoState<PointT> *state,
               Eigen::Ref<Eigen::Matrix4f> transform) {
    transform.setIdentity();
    transform.block<3,3>(0,0) = R;
    transform.block<3,1>(0,3) = T;

    cudaMemcpy(state->R_gpu, &R, sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice);
    cudaMemcpy(state->T_gpu, &T, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
}

template <typename PointT>
CvoResultInfo CvoGPU<PointT>::align(const PointCloud &source,
                                    const PointCloud &target,
                                    const Eigen::Matrix4f &T_init,
                                    bool return_association) const {
    CvoResultInfo result;
    if (source.empty() || target.empty()) {
        result.return_code = -1;
        return result;
    }

    // Convert to device vectors
    auto source_gpu = std::make_shared<CvoPointCloud<PointT>>(source); // copy
    auto target_gpu = std::make_shared<CvoPointCloud<PointT>>(target);
    CvoState<PointT> state(source_gpu, target_gpu, params_);

    Eigen::Matrix4f T = T_init; // T * p_target = p_source
    Eigen::Matrix3f R = T.block<3,3>(0,0);
    Eigen::Vector3f t = T.block<3,1>(0,3);

    int num_neighbors = params_.is_using_kdtree ? cvo::KDTREE_K_SIZE : params_.nearest_neighbors_max;
    std::queue<float> indicator_start_queue, indicator_end_queue;
    float indicator_start_sum = 0, indicator_end_sum = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < params_.MAX_ITER; ++iter) {
        state.reset_state_at_new_iter(num_neighbors);
        update_tf(R, t, &state, result.T_s2t); // result.T_s2t will be updated

        // Transform moving points
        transform_pointcloud_kernel<PointT><<< (state.num_moving + 255)/256, 256 >>>(
            thrust::raw_pointer_cast(state.cloud_y_gpu_init.data()),
            thrust::raw_pointer_cast(state.cloud_y_gpu.data()),
            state.num_moving, state.R_gpu, state.T_gpu);

        // Build kernel matrix
        fill_in_A_mat_gpu<PointT><<< (state.num_fixed + 255)/256, 256 >>>(
            params_gpu_,
            thrust::raw_pointer_cast(state.cloud_x_gpu.data()), state.num_fixed,
            thrust::raw_pointer_cast(state.cloud_y_gpu.data()), state.num_moving,
            num_neighbors, state.ell, state.A);
        cudaDeviceSynchronize();
        compute_nonzeros(&state.A_host);

        // Compute gradient
        compute_flow_gpu<PointT><<< (state.num_fixed + 255)/256, 256 >>>(
            params_gpu_,
            thrust::raw_pointer_cast(state.cloud_x_gpu.data()),
            thrust::raw_pointer_cast(state.cloud_y_gpu.data()),
            state.A, num_neighbors,
            thrust::raw_pointer_cast(state.omega_gpu.data()),
            thrust::raw_pointer_cast(state.v_gpu.data()));
        cudaDeviceSynchronize();

        Eigen::Vector3d omega_init(0.0, 0.0, 0.0);
        Eigen::Vector3d v_init(0.0, 0.0, 0.0);
        Eigen::Vector3d omega_d = thrust::reduce(state.omega_gpu.begin(), state.omega_gpu.end(), omega_init);
        Eigen::Vector3d v_d = thrust::reduce(state.v_gpu.begin(), state.v_gpu.end(), v_init);
        Eigen::Vector3f omega = omega_d.cast<float>();
        Eigen::Vector3f v = v_d.cast<float>();

        // Normalize gradient
        Eigen::Matrix<float,6,1> xi;
        xi << omega, v;
        xi.normalize();
        omega = xi.head<3>();
        v = xi.tail<3>();

        // Stop if gradient is small
        if (omega.norm() < params_.eps && v.norm() < params_.eps) {
            result.return_code = 0;
            break;
        }

        // Compute step size (simplified: use min_step for brevity; full polynomial solver can be added)
        float step = params_.min_step;

        // Update transformation: T = Exp(step * xi) * T
        Eigen::Matrix<float,6,1> xi_step = step * xi;
        Eigen::Matrix<float,3,4> dT = Exp_SE3(xi_step, true);
        Eigen::Matrix3f dR = dT.block<3,3>(0,0);
        Eigen::Vector3f dt = dT.block<3,1>(0,3);
        R = dR * R;
        t = dR * t + dt;

        // Check SE(3) distance
        Eigen::Matrix4f dT4 = Eigen::Matrix4f::Identity();
        dT4.block<3,3>(0,0) = dR;
        dT4.block<3,1>(0,3) = dt;
        double dist = dist_se3<float>(dR, dt);
        if (dist < params_.eps_2) {
            result.return_code = 0;
            break;
        }

        // Update ell (simplified decay)
        if (iter > params_.ell_decay_start) {
            state.ell *= params_.ell_decay_rate;
            if (state.ell < params_.ell_min) state.ell = params_.ell_min;
        }

        result.num_iters = iter + 1;
    }

    update_tf(R, t, &state, result.T_s2t);
    auto end_time = std::chrono::high_resolution_clock::now();
    result.registration_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (return_association) {
        gpu_association_to_cpu(state.A_host, result.association,
                               state.num_fixed, state.num_moving, num_neighbors);
    }
    return result;
}

template <typename PointT>
float CvoGPU<PointT>::function_angle(const PointCloud &source,
                                     const PointCloud &target,
                                     const Eigen::Matrix4f &T,
                                     float ell,
                                     bool approximate) const {
    (void)approximate;
    if (source.empty() || target.empty()) {
        return 0.0f;
    }
    float value = 0.0f;

    for (const auto& ps : source.points()) {
        for (const auto& pt : target.points()) {
            PointT moved = pt;
            const Eigen::Vector4f p_target_h(pt.x, pt.y, pt.z, 1.0f);
            const Eigen::Vector3f p_target = (T * p_target_h).head<3>();
            moved.x = p_target.x();
            moved.y = p_target.y();
            moved.z = p_target.z();
            value += detail::combined_kernel_weight(params_, ps, moved, ell);
        }
    }
    return value;
}

namespace detail {

constexpr int kMultiframeLinearSystemCpu = 0;
constexpr int kMultiframeLinearSystemGpu = 1;
constexpr int kMultiframeLinearSystemSparseCpu = 2;
constexpr int kMultiframeLinearSystemSparseGpu = 3;
constexpr int kMultiframeObjectiveEvalCpu = 0;
constexpr int kMultiframeObjectiveEvalGpu = 1;

template <typename PointT>
Eigen::Vector3d point_to_vec3(const PointT& point) {
    return Eigen::Vector3d(point.x, point.y, point.z);
}

template <typename PointT>
Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose_matrix(const CvoFrameGPU<PointT>& frame) {
    return Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame.pose_vec);
}

template <typename PointT>
Eigen::Vector3d transform_point(const CvoFrameGPU<PointT>& frame, const PointT& point) {
    const auto T = pose_matrix(frame);
    return T.template block<3,3>(0,0) * point_to_vec3(point) + T.template block<3,1>(0,3);
}

template <typename PointT>
Eigen::Matrix<double, 3, 6> right_perturbation_jacobian(const CvoFrameGPU<PointT>& frame, const PointT& point) {
    const auto T = pose_matrix(frame);
    const Eigen::Matrix3d R = T.template block<3,3>(0,0);
    const Eigen::Vector3d p = point_to_vec3(point);

    Eigen::Matrix<double, 3, 6> J;
    J.template leftCols<3>() = -R * skew<double>(p);
    J.template rightCols<3>() = R;
    return J;
}

template <typename PointT>
void apply_right_increment(CvoFrameGPU<PointT>& frame, const Eigen::Matrix<double, 6, 1>& delta) {
    const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> T_current = pose_matrix(frame);
    Eigen::Matrix4d T4 = Eigen::Matrix4d::Identity();
    T4.block<3,4>(0,0) = T_current;

    Eigen::Matrix4d delta4 = Eigen::Matrix4d::Identity();
    delta4.block<3,4>(0,0) = Exp_SE3(delta, true);

    const Eigen::Matrix4d updated = T4 * delta4;
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame.pose_vec) = updated.block<3,4>(0,0);
}

template <typename PointT>
double multiframe_weighted_residual_cost(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states) {
    (void)frames;
    double cost = 0.0;
    for (const auto& es : edge_states) {
        const SparseKernelMat64& A = es->get_A_cpu();
        const auto& pc1 = es->frame1()->points->points();
        const auto& pc2 = es->frame2()->points->points();
        const int nn = es->get_num_neighbors();
        for (int r = 0; r < A.rows; ++r) {
            for (int c = 0; c < nn; ++c) {
                const int idx2 = A.ind_row2col[r * nn + c];
                if (idx2 == -1) break;
                const double w = A.mat[r * nn + c];
                if (w <= 0.0) continue;
                const Eigen::Vector3d p1 = transform_point(*es->frame1(), pc1[r]);
                const Eigen::Vector3d p2 = transform_point(*es->frame2(), pc2[idx2]);
                cost += w * (p1 - p2).squaredNorm();
            }
        }
    }
    return cost;
}

template <typename PointT>
double multiframe_angle_objective(
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states) {
    double angle = 0.0;
    for (const auto& es : edge_states) {
        const SparseKernelMat64& A = es->get_A_cpu();
        angle += static_cast<double>(A_sum(const_cast<SparseKernelMat64*>(&A), es->get_num_neighbors()));
    }
    return angle;
}

} // namespace detail

namespace detail {

__device__ inline int dense_index_col_major(int row, int col, int nrows) {
    return row + col * nrows;
}

__device__ inline void load_pose_rotation_and_translation(const float* pose,
                                                          double R[9],
                                                          double t[3]) {
    R[0] = pose[0];  R[1] = pose[1];  R[2] = pose[2];
    R[3] = pose[4];  R[4] = pose[5];  R[5] = pose[6];
    R[6] = pose[8];  R[7] = pose[9];  R[8] = pose[10];
    t[0] = pose[3];
    t[1] = pose[7];
    t[2] = pose[11];
}

__device__ inline void mat3_vec3(const double R[9], double x, double y, double z, double out[3]) {
    out[0] = R[0] * x + R[1] * y + R[2] * z;
    out[1] = R[3] * x + R[4] * y + R[5] * z;
    out[2] = R[6] * x + R[7] * y + R[8] * z;
}

template <typename PointT>
__global__ void assemble_edge_linear_system_kernel(
    const PointT* points1_init,
    const PointT* points2_init,
    const float* pose1,
    const float* pose2,
    const SparseKernelMat64* A_selected,
    int num_neighbors,
    int frame_idx_a,
    int frame_idx_b,
    int system_size,
    double* H,
    double* g) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = A_selected->rows * num_neighbors;
    if (tid >= total_entries) return;

    const int r = tid / num_neighbors;
    const int c = tid % num_neighbors;
    if (c >= static_cast<int>(A_selected->nonzeros[r])) return;

    const int idx2 = A_selected->ind_row2col[tid];
    if (idx2 < 0) return;
    const double w = static_cast<double>(A_selected->mat[tid]);
    if (!(w > 0.0)) return;

    const PointT p1 = points1_init[r];
    const PointT p2 = points2_init[idx2];

    double R1[9], t1[3], R2[9], t2[3];
    load_pose_rotation_and_translation(pose1, R1, t1);
    load_pose_rotation_and_translation(pose2, R2, t2);

    double Rp1[3], Rp2[3];
    mat3_vec3(R1, static_cast<double>(p1.x), static_cast<double>(p1.y), static_cast<double>(p1.z), Rp1);
    mat3_vec3(R2, static_cast<double>(p2.x), static_cast<double>(p2.y), static_cast<double>(p2.z), Rp2);

    const double r_vec[3] = {
        (Rp1[0] + t1[0]) - (Rp2[0] + t2[0]),
        (Rp1[1] + t1[1]) - (Rp2[1] + t2[1]),
        (Rp1[2] + t1[2]) - (Rp2[2] + t2[2])
    };

    const double px1 = static_cast<double>(p1.x);
    const double py1 = static_cast<double>(p1.y);
    const double pz1 = static_cast<double>(p1.z);
    const double px2 = static_cast<double>(p2.x);
    const double py2 = static_cast<double>(p2.y);
    const double pz2 = static_cast<double>(p2.z);

    const double skew1[9] = {0.0, -pz1, py1, pz1, 0.0, -px1, -py1, px1, 0.0};
    const double skew2[9] = {0.0, -pz2, py2, pz2, 0.0, -px2, -py2, px2, 0.0};

    double J1[18];
    double J2[18];
    for (int row = 0; row < 3; ++row) {
        const int rbase = row * 6;
        const int Rbase = row * 3;
        for (int col = 0; col < 3; ++col) {
            double rot1 = 0.0;
            double rot2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                rot1 += R1[Rbase + k] * skew1[k * 3 + col];
                rot2 += R2[Rbase + k] * skew2[k * 3 + col];
            }
            J1[rbase + col] = -rot1;
            J1[rbase + 3 + col] = R1[Rbase + col];
            J2[rbase + col] = rot2;
            J2[rbase + 3 + col] = -R2[Rbase + col];
        }
    }

    const double sqrt_w = sqrt(w);
    const int base_a = frame_idx_a * 6;
    const int base_b = frame_idx_b * 6;

    double Jr1[6] = {0, 0, 0, 0, 0, 0};
    double Jr2[6] = {0, 0, 0, 0, 0, 0};
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 3; ++row) {
            const double rw = sqrt_w * r_vec[row];
            Jr1[col] += (sqrt_w * J1[row * 6 + col]) * rw;
            Jr2[col] += (sqrt_w * J2[row * 6 + col]) * rw;
        }
    }

    for (int i = 0; i < 6; ++i) {
        atomicAdd(&g[base_a + i], Jr1[i]);
        atomicAdd(&g[base_b + i], Jr2[i]);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            double haa = 0.0;
            double hbb = 0.0;
            double hab = 0.0;
            for (int row = 0; row < 3; ++row) {
                const double j1i = sqrt_w * J1[row * 6 + i];
                const double j1j = sqrt_w * J1[row * 6 + j];
                const double j2i = sqrt_w * J2[row * 6 + i];
                const double j2j = sqrt_w * J2[row * 6 + j];
                haa += j1i * j1j;
                hbb += j2i * j2j;
                hab += j1i * j2j;
            }
            atomicAdd(&H[dense_index_col_major(base_a + i, base_a + j, system_size)], haa);
            atomicAdd(&H[dense_index_col_major(base_b + i, base_b + j, system_size)], hbb);
            atomicAdd(&H[dense_index_col_major(base_a + i, base_b + j, system_size)], hab);
            atomicAdd(&H[dense_index_col_major(base_b + j, base_a + i, system_size)], hab);
        }
    }
}

template <typename PointT>
__global__ void accumulate_edge_weighted_residual_cost_kernel(
    const PointT* points1_init,
    const PointT* points2_init,
    const float* pose1,
    const float* pose2,
    const SparseKernelMat64* A_selected,
    int num_neighbors,
    double* cost_out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = A_selected->rows * num_neighbors;
    if (tid >= total_entries) return;

    const int r = tid / num_neighbors;
    const int c = tid % num_neighbors;
    if (c >= static_cast<int>(A_selected->nonzeros[r])) return;

    const int idx2 = A_selected->ind_row2col[tid];
    if (idx2 < 0) return;
    const double w = static_cast<double>(A_selected->mat[tid]);
    if (!(w > 0.0)) return;

    const PointT p1 = points1_init[r];
    const PointT p2 = points2_init[idx2];

    double R1[9], t1[3], R2[9], t2[3];
    load_pose_rotation_and_translation(pose1, R1, t1);
    load_pose_rotation_and_translation(pose2, R2, t2);

    double Rp1[3], Rp2[3];
    mat3_vec3(R1, static_cast<double>(p1.x), static_cast<double>(p1.y), static_cast<double>(p1.z), Rp1);
    mat3_vec3(R2, static_cast<double>(p2.x), static_cast<double>(p2.y), static_cast<double>(p2.z), Rp2);

    const double dx = (Rp1[0] + t1[0]) - (Rp2[0] + t2[0]);
    const double dy = (Rp1[1] + t1[1]) - (Rp2[1] + t2[1]);
    const double dz = (Rp1[2] + t1[2]) - (Rp2[2] + t2[2]);
    atomicAdd(cost_out, w * (dx * dx + dy * dy + dz * dz));
}

} // namespace detail

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------

template <typename PointT>
__global__ void transform_pointcloud_kernel(const PointT *input, PointT *output,
                                            int num_points,
                                            const Eigen::Matrix3f *R, const Eigen::Vector3f *T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    PointT p = input[i];
    Eigen::Vector3f pos(p.x, p.y, p.z);
    Eigen::Vector3f trans = (*R) * pos + (*T);
    p.x = trans.x(); p.y = trans.y(); p.z = trans.z();
    output[i] = p;
}

template <typename PointT>
__global__ void fill_in_A_mat_gpu(const CvoParams *params,
                                  const PointT *points_a, int a_size,
                                  const PointT *points_b, int b_size,
                                  int num_neighbors, float ell,
                                  SparseKernelMat *A_mat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a_size) return;

    const PointT &pa = points_a[i];
    unsigned int count = 0;
    for (int j = 0; j < b_size && count < num_neighbors; ++j) {
        const PointT &pb = points_b[j];
        const float k = detail::combined_kernel_weight(*params, pa, pb, ell);
        if (k > 0.0f) {
            A_mat->mat[i * num_neighbors + count] = k;
            A_mat->ind_row2col[i * num_neighbors + count] = j;
            ++count;
        }
    }
    A_mat->nonzeros[i] = count;
}

template <typename PointT>
__global__ void compute_flow_gpu(const CvoParams *params,
                                 const PointT *cloud_x, const PointT *cloud_y,
                                 const SparseKernelMat *A, int num_neighbors,
                                 Eigen::Vector3d *omega_gpu, Eigen::Vector3d *v_gpu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A->rows) return;

    Eigen::Vector3f px(cloud_x[i].x, cloud_x[i].y, cloud_x[i].z);
    Eigen::Vector3f omega_i = Eigen::Vector3f::Zero();
    Eigen::Vector3f v_i = Eigen::Vector3f::Zero();

    for (int j = 0; j < num_neighbors; ++j) {
        int idx = A->ind_row2col[i * num_neighbors + j];
        if (idx == -1) break;
        const PointT &py = cloud_y[idx];
        Eigen::Vector3f p_y(py.x, py.y, py.z);
        float a_ij = A->mat[i * num_neighbors + j];
        omega_i += a_ij * px.cross(p_y);
        v_i += a_ij * (p_y - px);
    }
    omega_gpu[i] = (omega_i / params->c).cast<double>();
    v_gpu[i] = (v_i / params->d).cast<double>();
}

// -----------------------------------------------------------------------------
// Implementation (multi‑frame part)
// -----------------------------------------------------------------------------

template <typename PointT>
int CvoGPU<PointT>::align_multiframe(
    std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
    const std::vector<bool>& fixed_flags,
    double* registration_seconds) {

    const size_t N = frames.size();
    const int pose_dof = 6;
    double ell = params_.multiframe_ell_init;
    std::ofstream iter_log;
    std::ofstream pose_log;
    if (params_.multiframe_enable_iteration_log && !params_.multiframe_iteration_log_path.empty()) {
        iter_log.open(params_.multiframe_iteration_log_path, std::ios::out | std::ios::trunc);
        if (iter_log.is_open()) {
            iter_log << "ell_level,iter_in_ell,ell,min_nonzeros,baseline_cost,trial_cost,"
                     << "baseline_angle,trial_angle,raw_dx_norm,accepted_dx_norm,accepted,line_search_steps,"
                     << "transform_ms,update_inner_product_ms,linear_system_ms,solve_ms,"
                     << "objective_eval_ms,line_search_ms,iter_total_ms\n";
        }
    }
    if (params_.multiframe_enable_pose_log && !params_.multiframe_pose_log_path.empty()) {
        pose_log.open(params_.multiframe_pose_log_path, std::ios::out | std::ios::trunc);
        if (pose_log.is_open()) {
            pose_log << "ell_level,iter_in_ell,ell,frame_id,"
                     << "r00,r01,r02,t0,r10,r11,r12,t1,r20,r21,r22,t2\n";
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    last_multiframe_debug_ = MultiframeDebugInfo{};

    if (frames.empty()) {
        if (registration_seconds) {
            *registration_seconds = 0.0;
        }
        return -1;
    }

    std::cout << "[CvoGPU] align_multiframe start:"
              << " frames=" << N
              << " edges=" << edge_states.size()
              << " linear_backend=" << params_.multiframe_linear_system_backend
              << " sparse_fill_backend=" << params_.multiframe_sparse_fill_backend
              << " objective_backend=" << params_.multiframe_objective_eval_backend
              << " line_search=" << params_.multiframe_enable_line_search
              << " stream_frame_clouds_gpu=" << params_.multiframe_stream_frame_clouds_gpu
              << " ell_init=" << params_.multiframe_ell_init
              << " ell_min=" << params_.multiframe_ell_min
              << " iters_per_ell=" << params_.multiframe_iterations_per_ell
              << " max_ell_levels=" << params_.multiframe_max_iters
              << std::endl;

    for (int outer = 0; outer < params_.multiframe_max_iters && ell >= params_.multiframe_ell_min; ++outer) {
        for (auto& es : edge_states) es->set_ell(ell);
        std::cout << "[CvoGPU] ell_level=" << outer
                  << " ell=" << ell
                  << " begin" << std::endl;

        bool ell_converged = false;
        for (int inner = 0; inner < params_.multiframe_iterations_per_ell; ++inner) {
            const auto iter_start_time = std::chrono::high_resolution_clock::now();
            const bool use_sparse_cpu =
                params_.multiframe_linear_system_backend == detail::kMultiframeLinearSystemSparseCpu;
            const bool use_sparse_gpu =
                params_.multiframe_linear_system_backend == detail::kMultiframeLinearSystemSparseGpu;
            const bool stream_sparse_edges =
                (use_sparse_cpu || use_sparse_gpu) &&
                params_.multiframe_release_binary_state_gpu_each_iter != 0;
            const bool stream_frame_clouds =
                stream_sparse_edges && params_.multiframe_stream_frame_clouds_gpu != 0;

            std::uint64_t min_nz = std::numeric_limits<std::uint64_t>::max();
            double transform_ms = 0.0;
            double update_inner_product_ms = 0.0;
            if (stream_frame_clouds) {
                const auto frame_to_index = sparse_pose::build_frame_index_map(frames);
                std::vector<int> frame_use_counts(frames.size(), 0);
                for (const auto& es : edge_states) {
                    const int a = frame_to_index.at(es->frame1().get());
                    const int b = frame_to_index.at(es->frame2().get());
                    ++frame_use_counts[a];
                    if (b != a) {
                        ++frame_use_counts[b];
                    }
                }
                std::vector<char> frame_ready(frames.size(), 0);
                for (const auto& es : edge_states) {
                    const int a = frame_to_index.at(es->frame1().get());
                    const int b = frame_to_index.at(es->frame2().get());
                    const auto transform_start_time = std::chrono::high_resolution_clock::now();
                    if (!frame_ready[a]) {
                        frames[a]->transform_pointcloud();
                        frame_ready[a] = 1;
                    }
                    if (b != a && !frame_ready[b]) {
                        frames[b]->transform_pointcloud();
                        frame_ready[b] = 1;
                    }
                    transform_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - transform_start_time).count();

                    const auto update_inner_product_start_time = std::chrono::high_resolution_clock::now();
                    const std::uint64_t nz = es->update_inner_product();
                    min_nz = std::min(min_nz, nz);
                    es->free_gpu_state_memory();
                    update_inner_product_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - update_inner_product_start_time).count();

                    if (--frame_use_counts[a] == 0 && frame_ready[a]) {
                        frames[a]->release_gpu_clouds();
                        frame_ready[a] = 0;
                    }
                    if (b != a && --frame_use_counts[b] == 0 && frame_ready[b]) {
                        frames[b]->release_gpu_clouds();
                        frame_ready[b] = 0;
                    }
                }
                for (size_t i = 0; i < frames.size(); ++i) {
                    if (frame_ready[i]) {
                        frames[i]->release_gpu_clouds();
                    }
                }
            } else {
                const auto transform_start_time = std::chrono::high_resolution_clock::now();
                for (auto& f : frames) f->transform_pointcloud();
                transform_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - transform_start_time).count();

                const auto update_inner_product_start_time = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for reduction(min:min_nz)
                for (size_t i = 0; i < edge_states.size(); ++i) {
                    const std::uint64_t nz = edge_states[i]->update_inner_product();
                    min_nz = std::min(min_nz, nz);
                }
                update_inner_product_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - update_inner_product_start_time).count();
            }
            if (min_nz < static_cast<std::uint64_t>(params_.multiframe_min_nonzeros)) {
                std::cout << "[CvoGPU] ell_level=" << outer
                          << " iter=" << inner
                          << " stopped: min_nonzeros=" << min_nz
                          << " < " << params_.multiframe_min_nonzeros << std::endl;
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                if (iter_log.is_open()) {
                    iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz
                             << ",nan,nan,nan,nan,nan,nan,0,0,"
                             << transform_ms << ',' << update_inner_product_ms
                             << ",nan,nan,nan,nan," << iter_total_ms << '\n';
                }
                ell_converged = true;
                break;
            }

            const auto linear_system_start_time = std::chrono::high_resolution_clock::now();
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N * pose_dof, N * pose_dof);
            Eigen::VectorXd g = Eigen::VectorXd::Zero(N * pose_dof);
            Eigen::VectorXd dx;
            bool solve_ok = true;
            if (use_sparse_cpu || use_sparse_gpu) {
                BlockSparsePoseSystem sparse_system;
                if (stream_sparse_edges) {
                    const auto frame_to_index = sparse_pose::build_frame_index_map(frames);
                    std::vector<std::pair<int, int>> sparse_edges;
                    sparse_edges.reserve(edge_states.size());
                    for (const auto& es : edge_states) {
                        sparse_edges.emplace_back(
                            frame_to_index.at(es->frame1().get()),
                            frame_to_index.at(es->frame2().get()));
                    }
                    sparse_pose::initialize_block_sparse_pose_system(
                        static_cast<int>(frames.size()), sparse_edges, sparse_system);
                    for (const auto& es : edge_states) {
                        const int a = frame_to_index.at(es->frame1().get());
                        const int b = frame_to_index.at(es->frame2().get());
                        sparse_pose::accumulate_block_sparse_pose_edge(
                            frames[a], frames[b], a, b, es->get_A_cpu(), es->get_num_neighbors(),
                            [](const auto& frame, const auto& point) {
                                return detail::transform_point(frame, point);
                            },
                            [](const auto& frame, const auto& point) {
                                return detail::right_perturbation_jacobian(frame, point);
                            },
                            sparse_system);
                        es->free_gpu_state_memory();
                    }
                    sparse_pose::finalize_block_sparse_pose_system(fixed_flags, sparse_system);
                } else {
                    build_block_sparse_system(frames, edge_states, fixed_flags, sparse_system);
                }
                const std::size_t unknowns = static_cast<std::size_t>(sparse_system.num_poses) * pose_dof;
                const std::size_t offdiag_blocks = sparse_system.offdiag_blocks.size();
                const std::size_t dense_doubles = unknowns * unknowns;
                const double dense_mb =
                    static_cast<double>(dense_doubles * sizeof(double)) / (1024.0 * 1024.0);
                std::cout << "[CvoGPU] sparse LS size:"
                          << " unknowns=" << unknowns
                          << " diag_blocks=" << sparse_system.diag_blocks.size()
                          << " offdiag_blocks=" << offdiag_blocks
                          << " dense_equiv_mb=" << dense_mb
                          << " solver=" << (use_sparse_gpu ? "gpu_sparse" : "cpu_sparse")
                          << " streaming=" << stream_sparse_edges
                          << std::endl;
                if (use_sparse_gpu) {
                    solve_ok = solve_sparse_block_system_gpu(sparse_system, dx);
                } else {
                    solve_ok = solve_sparse_block_system_cpu(sparse_system, dx);
                }
            } else {
                H.setZero(); g.setZero();
                if (params_.multiframe_linear_system_backend == detail::kMultiframeLinearSystemGpu) {
                    build_linear_system_gpu(frames, edge_states, fixed_flags, H, g);
                } else {
                    build_linear_system(frames, edge_states, fixed_flags, H, g);
                }
                H = 0.5 * (H + H.transpose());
            }
            const double linear_system_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - linear_system_start_time).count();

            const int debug_frame = static_cast<int>(std::min<std::size_t>(1, N - 1));
            const int debug_idx = debug_frame * pose_dof;
            Eigen::Matrix<double, 6, 1> g_frame_before_fix = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 6> H_frame_before_fix = Eigen::Matrix<double, 6, 6>::Zero();
            if (g.size() >= debug_idx + pose_dof) {
                g_frame_before_fix = g.segment(debug_idx, pose_dof);
            }
            if (H.rows() >= debug_idx + pose_dof) {
                H_frame_before_fix = H.block(debug_idx, debug_idx, pose_dof, pose_dof);
            }

            double solve_ms = 0.0;
            if (!use_sparse_cpu && !use_sparse_gpu) {
                for (size_t i = 0; i < N; ++i) {
                    if (fixed_flags[i]) {
                        const int idx = static_cast<int>(i) * pose_dof;
                        H.block(idx, 0, pose_dof, H.cols()).setZero();
                        H.block(0, idx, H.rows(), pose_dof).setZero();
                        H.block(idx, idx, pose_dof, pose_dof).setIdentity();
                        g.segment(idx, pose_dof).setZero();
                    } else {
                        const int idx = static_cast<int>(i) * pose_dof;
                        H.block(idx, idx, pose_dof, pose_dof).diagonal().array() += 1e-6;
                    }
                }

                const auto solve_start_time = std::chrono::high_resolution_clock::now();
                dx = H.ldlt().solve(-g);
                solve_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - solve_start_time).count();
            }
            if (!solve_ok || !dx.allFinite()) {
                std::cout << "[CvoGPU] ell_level=" << outer
                          << " iter=" << inner
                          << " solve failed or non-finite step" << std::endl;
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                if (iter_log.is_open()) {
                    iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz
                             << ",nan,nan,nan,nan,nan,nan,0,0,"
                             << transform_ms << ',' << update_inner_product_ms << ','
                             << linear_system_ms << ',' << solve_ms
                             << ",nan,nan," << iter_total_ms << '\n';
                }
                ell_converged = true;
                break;
            }

            const double dx_norm = dx.norm();
            if (!std::isfinite(dx_norm)) {
                std::cout << "[CvoGPU] ell_level=" << outer
                          << " iter=" << inner
                          << " invalid step norm" << std::endl;
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                if (iter_log.is_open()) {
                    iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz
                             << ",nan,nan,nan,nan,nan,nan,0,0,"
                             << transform_ms << ',' << update_inner_product_ms << ','
                             << linear_system_ms << ',' << solve_ms
                             << ",nan,nan," << iter_total_ms << '\n';
                }
                ell_converged = true;
                break;
            }
            if (dx_norm < 1e-5) {
                std::cout << "[CvoGPU] ell_level=" << outer
                          << " iter=" << inner
                          << " converged: raw_dx_norm=" << dx_norm << std::endl;
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                if (iter_log.is_open()) {
                    iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz
                             << ",nan,nan,nan,nan," << dx_norm << ',' << dx_norm << ",1,0,"
                             << transform_ms << ',' << update_inner_product_ms << ','
                             << linear_system_ms << ',' << solve_ms
                             << ",nan,nan," << iter_total_ms << '\n';
                }
                ell_converged = true;
                break;
            }

            if (dx_norm > 0.25) {
                dx *= (0.25 / dx_norm);
            }

            const bool enable_line_search = params_.multiframe_enable_line_search != 0;
            const bool release_binary_state_gpu = params_.multiframe_release_binary_state_gpu_each_iter != 0;
            const bool use_gpu_objective_eval =
                params_.multiframe_objective_eval_backend == detail::kMultiframeObjectiveEvalGpu &&
                !(release_binary_state_gpu && (use_sparse_cpu || use_sparse_gpu));
            const bool need_objective_eval = enable_line_search || iter_log.is_open();
            if (params_.multiframe_objective_eval_backend == detail::kMultiframeObjectiveEvalGpu &&
                !use_gpu_objective_eval) {
                std::cout << "[CvoGPU] falling back to CPU objective evaluation because GPU binary-state buffers are released in sparse mode"
                          << std::endl;
            }

            auto evaluate_cost = [&]() -> double {
                if (use_gpu_objective_eval) {
                    return evaluate_multiframe_weighted_residual_cost_gpu(frames, edge_states);
                }
                return detail::multiframe_weighted_residual_cost(frames, edge_states);
            };
            auto evaluate_angle = [&]() -> double {
                if (use_gpu_objective_eval) {
                    return evaluate_multiframe_angle_objective_gpu(edge_states);
                }
                return detail::multiframe_angle_objective(edge_states);
            };

            double baseline_cost = std::numeric_limits<double>::quiet_NaN();
            double baseline_angle = std::numeric_limits<double>::quiet_NaN();
            const auto objective_eval_start_time = std::chrono::high_resolution_clock::now();
            if (need_objective_eval) {
                baseline_cost = evaluate_cost();
                baseline_angle = evaluate_angle();
            }
            const double objective_eval_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - objective_eval_start_time).count();
            std::vector<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> saved_poses;
            saved_poses.reserve(frames.size());
            for (const auto& frame : frames) {
                saved_poses.push_back(Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame->pose_vec));
            }

            bool accepted = false;
            double step_scale = 1.0;
            int line_search_steps = 0;
            double accepted_cost = std::numeric_limits<double>::quiet_NaN();
            double accepted_angle = std::numeric_limits<double>::quiet_NaN();
            double accepted_step_norm = std::numeric_limits<double>::quiet_NaN();
            const auto line_search_start_time = std::chrono::high_resolution_clock::now();
            if (!enable_line_search) {
                for (size_t i = 0; i < N; ++i) {
                    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frames[i]->pose_vec) = saved_poses[i];
                    if (fixed_flags[i]) continue;
                    detail::apply_right_increment(*frames[i], dx.segment<6>(static_cast<int>(i) * pose_dof));
                    if (use_gpu_objective_eval) {
                        frames[i]->sync_pose_to_gpu();
                    }
                }
                line_search_steps = 0;
                accepted_step_norm = dx.norm();
                accepted = true;
                if (need_objective_eval) {
                    accepted_cost = evaluate_cost();
                    accepted_angle = baseline_angle;
                }
            } else {
                for (int ls = 0; ls < 12; ++ls) {
                    line_search_steps = ls + 1;
                    for (size_t i = 0; i < N; ++i) {
                        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frames[i]->pose_vec) = saved_poses[i];
                        if (fixed_flags[i]) continue;
                        detail::apply_right_increment(*frames[i], step_scale * dx.segment<6>(static_cast<int>(i) * pose_dof));
                        if (use_gpu_objective_eval) {
                            frames[i]->sync_pose_to_gpu();
                        }
                    }

                    const double trial_cost = evaluate_cost();
                    const double trial_angle = baseline_angle;
                    if (std::isfinite(trial_cost) && std::isfinite(trial_angle) &&
                        trial_cost <= baseline_cost &&
                        trial_angle + 1e-8 >= baseline_angle) {
                        accepted_cost = trial_cost;
                        accepted_angle = trial_angle;
                        accepted_step_norm = (step_scale * dx).norm();
                        accepted = true;
                        break;
                    }
                    step_scale *= 0.5;
                }
            }
            const double line_search_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - line_search_start_time).count();

            if (!accepted) {
                for (size_t i = 0; i < N; ++i) {
                    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frames[i]->pose_vec) = saved_poses[i];
                }
                std::cout << "[CvoGPU] ell_level=" << outer
                          << " iter=" << inner
                          << " rejected all trial steps" << std::endl;
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                if (iter_log.is_open()) {
                    iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz << ','
                             << baseline_cost << ",nan," << baseline_angle << ",nan,"
                             << dx_norm << ",nan,0," << line_search_steps << ','
                             << transform_ms << ',' << update_inner_product_ms << ','
                             << linear_system_ms << ',' << solve_ms << ','
                             << objective_eval_ms << ',' << line_search_ms << ','
                             << iter_total_ms << '\n';
                }
                ell_converged = true;
                break;
            }

            const double trial_norm = (step_scale * dx).norm();
            if (!last_multiframe_debug_.valid) {
                last_multiframe_debug_.valid = true;
                last_multiframe_debug_.ell = ell;
                last_multiframe_debug_.ell_level = outer;
                last_multiframe_debug_.iter_in_ell = inner;
                last_multiframe_debug_.min_nonzeros = min_nz;
                last_multiframe_debug_.baseline_cost = baseline_cost;
                last_multiframe_debug_.baseline_angle = baseline_angle;
                last_multiframe_debug_.gradient_frame = g_frame_before_fix;
                last_multiframe_debug_.gradient_norm = g_frame_before_fix.norm();
                last_multiframe_debug_.hessian_block = H_frame_before_fix;
                last_multiframe_debug_.raw_step =
                    dx.segment(debug_idx, pose_dof);
                last_multiframe_debug_.raw_step_norm =
                    last_multiframe_debug_.raw_step.norm();
                last_multiframe_debug_.accepted_step =
                    step_scale * dx.segment(debug_idx, pose_dof);
                last_multiframe_debug_.accepted_step_norm =
                    std::isfinite(accepted_step_norm) ? accepted_step_norm : trial_norm;
            }
            if (iter_log.is_open()) {
                const double iter_total_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - iter_start_time).count();
                iter_log << outer << ',' << inner << ',' << ell << ',' << min_nz << ','
                         << baseline_cost << ',' << accepted_cost << ','
                         << baseline_angle << ',' << accepted_angle << ','
                         << dx_norm << ',' << trial_norm << ",1," << line_search_steps << ','
                         << transform_ms << ',' << update_inner_product_ms << ','
                         << linear_system_ms << ',' << solve_ms << ','
                         << objective_eval_ms << ',' << line_search_ms << ','
                         << iter_total_ms << '\n';
            }
            const double iter_total_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - iter_start_time).count();
            std::cout << "[CvoGPU] ell_level=" << outer
                      << " iter=" << inner
                      << " accepted"
                      << " min_nz=" << min_nz
                      << " cost=" << accepted_cost
                      << " angle=" << accepted_angle
                      << " step_norm=" << accepted_step_norm
                      << " ls_steps=" << line_search_steps
                      << " t_ms(update=" << update_inner_product_ms
                      << ",linear=" << linear_system_ms
                      << ",obj=" << objective_eval_ms
                      << ",ls=" << line_search_ms
                      << ",total=" << iter_total_ms << ")"
                      << std::endl;
            if (pose_log.is_open()) {
                for (size_t i = 0; i < N; ++i) {
                    const auto pose = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frames[i]->pose_vec);
                    pose_log << outer << ',' << inner << ',' << ell << ',' << i << ','
                             << pose(0,0) << ',' << pose(0,1) << ',' << pose(0,2) << ',' << pose(0,3) << ','
                             << pose(1,0) << ',' << pose(1,1) << ',' << pose(1,2) << ',' << pose(1,3) << ','
                             << pose(2,0) << ',' << pose(2,1) << ',' << pose(2,2) << ',' << pose(2,3) << '\n';
                }
            }
            if (params_.multiframe_release_binary_state_gpu_each_iter &&
                !use_sparse_cpu && !use_sparse_gpu) {
                for (auto& es : edge_states) {
                    es->free_gpu_state_memory();
                }
                if (params_.multiframe_linear_system_backend == detail::kMultiframeLinearSystemGpu ||
                    params_.multiframe_linear_system_backend == detail::kMultiframeLinearSystemSparseGpu ||
                    params_.multiframe_objective_eval_backend == detail::kMultiframeObjectiveEvalGpu) {
                    release_multiframe_system_buffers();
                }
                std::cout << "[CvoGPU] released BinaryStateGPU buffers after accepted iteration" << std::endl;
            }
            if (!std::isfinite(trial_norm) || trial_norm < 1e-3) {
                ell_converged = true;
                break;
            }
        }

        if (ell_converged) {
            if (ell <= params_.multiframe_ell_min + 1e-9) {
                std::cout << "[CvoGPU] final ell converged at ell=" << ell << std::endl;
                break;
            }
            for (auto& es : edge_states) es->update_ell();
            const double next_ell = edge_states[0]->get_ell();
            std::cout << "[CvoGPU] ell_level=" << outer
                      << " converged, decaying ell from " << ell
                      << " to " << next_ell << std::endl;
            ell = edge_states[0]->get_ell();
        }
    }

    if (registration_seconds) {
        auto end_time = std::chrono::high_resolution_clock::now();
        *registration_seconds = std::chrono::duration<double>(end_time - start_time).count();
    }
    if (params_.multiframe_release_binary_state_gpu_each_iter) {
        for (auto& es : edge_states) {
            es->free_state_memory();
        }
        release_multiframe_system_buffers();
    }
    if (params_.multiframe_stream_frame_clouds_gpu) {
        for (auto& f : frames) {
            f->release_gpu_clouds();
        }
    }
    std::cout << "[CvoGPU] align_multiframe done"
              << " seconds="
              << std::chrono::duration<double>(
                     std::chrono::high_resolution_clock::now() - start_time).count()
              << std::endl;
    return 0;
}

template <typename PointT>
void CvoGPU<PointT>::build_linear_system_gpu(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
    const std::vector<bool>& fixed_flags,
    Eigen::MatrixXd& H,
    Eigen::VectorXd& g) {
    (void)fixed_flags;
    const int pose_dof = 6;
    const int system_size = static_cast<int>(frames.size()) * pose_dof;
    ensure_multiframe_system_buffers(system_size);
    cudaMemset(multiframe_H_device_, 0, static_cast<size_t>(system_size) * static_cast<size_t>(system_size) * sizeof(double));
    cudaMemset(multiframe_g_device_, 0, static_cast<size_t>(system_size) * sizeof(double));

    std::unordered_map<const CvoFrameGPU<PointT>*, int> frame_to_index;
    frame_to_index.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        frame_to_index.emplace(frames[i].get(), static_cast<int>(i));
    }

    const int threads = 256;
    for (const auto& es : edge_states) {
        const int a = frame_to_index.at(es->frame1().get());
        const int b = frame_to_index.at(es->frame2().get());
        const int nn = es->get_num_neighbors();
        const int total_entries = static_cast<int>(es->frame1()->size()) * nn;
        const int blocks = (total_entries + threads - 1) / threads;
        detail::assemble_edge_linear_system_kernel<PointT><<<blocks, threads>>>(
            thrust::raw_pointer_cast(es->frame1()->points_init_gpu()->points.data()),
            thrust::raw_pointer_cast(es->frame2()->points_init_gpu()->points.data()),
            es->frame1()->pose_vec_gpu(),
            es->frame2()->pose_vec_gpu(),
            es->get_A_gpu(),
            nn,
            a,
            b,
            system_size,
            multiframe_H_device_,
            multiframe_g_device_);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(H.data(), multiframe_H_device_,
               static_cast<size_t>(system_size) * static_cast<size_t>(system_size) * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(), multiframe_g_device_,
               static_cast<size_t>(system_size) * sizeof(double),
               cudaMemcpyDeviceToHost);
}

template <typename PointT>
double CvoGPU<PointT>::evaluate_multiframe_weighted_residual_cost_gpu(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states) {
    (void)frames;
    ensure_multiframe_system_buffers(static_cast<int>(frames.size()) * 6);
    cudaMemset(multiframe_cost_device_, 0, sizeof(double));

    const int threads = 256;
    for (const auto& es : edge_states) {
        const int nn = es->get_num_neighbors();
        const int total_entries = static_cast<int>(es->frame1()->size()) * nn;
        const int blocks = (total_entries + threads - 1) / threads;
        detail::accumulate_edge_weighted_residual_cost_kernel<PointT><<<blocks, threads>>>(
            thrust::raw_pointer_cast(es->frame1()->points_init_gpu()->points.data()),
            thrust::raw_pointer_cast(es->frame2()->points_init_gpu()->points.data()),
            es->frame1()->pose_vec_gpu(),
            es->frame2()->pose_vec_gpu(),
            es->get_A_gpu(),
            nn,
            multiframe_cost_device_);
    }
    cudaDeviceSynchronize();
    double cost = 0.0;
    cudaMemcpy(&cost, multiframe_cost_device_, sizeof(double), cudaMemcpyDeviceToHost);
    return cost;
}

template <typename PointT>
double CvoGPU<PointT>::evaluate_multiframe_angle_objective_gpu(
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states) {
    double angle = 0.0;
    for (const auto& es : edge_states) {
        angle += static_cast<double>(A_sum(const_cast<SparseKernelMat64*>(&es->get_A_gpu_host()), es->get_num_neighbors()));
    }
    return angle;
}

template <typename PointT>
void CvoGPU<PointT>::build_block_sparse_system(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
    const std::vector<bool>& fixed_flags,
    BlockSparsePoseSystem& system) {
    sparse_pose::build_block_sparse_pose_system(
        frames, edge_states, fixed_flags,
        [](const auto& frame, const auto& point) {
            return detail::transform_point(frame, point);
        },
        [](const auto& frame, const auto& point) {
            return detail::right_perturbation_jacobian(frame, point);
        },
        system);
}

template <typename PointT>
bool CvoGPU<PointT>::solve_sparse_block_system_cpu(
    const BlockSparsePoseSystem& system,
    Eigen::VectorXd& dx) {
    return cpu_sparse_pose_solver_.solve(system, dx);
}

template <typename PointT>
bool CvoGPU<PointT>::solve_sparse_block_system_gpu(
    const BlockSparsePoseSystem& system,
    Eigen::VectorXd& dx) {
    return gpu_sparse_pose_solver_.solve(system, dx);
}

template <typename PointT>
void CvoGPU<PointT>::build_linear_system(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
    const std::vector<bool>& fixed_flags,
    Eigen::MatrixXd& H,
    Eigen::VectorXd& g) {
    (void)fixed_flags;
    const int pose_dof = 6;
    std::unordered_map<const CvoFrameGPU<PointT>*, int> frame_to_index;
    frame_to_index.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        frame_to_index.emplace(frames[i].get(), static_cast<int>(i));
    }

    #pragma omp parallel for
    for (size_t e = 0; e < edge_states.size(); ++e) {
        auto& es = edge_states[e];
        const int a = frame_to_index.at(es->frame1().get());
        const int b = frame_to_index.at(es->frame2().get());
        const SparseKernelMat64& A = es->get_A_cpu();
        int nn = es->get_num_neighbors();

        const auto& pc1 = frames[a]->points->points();
        const auto& pc2 = frames[b]->points->points();

        Eigen::Matrix<double, 6, 6> H_aa = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> H_bb = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 6> H_ab = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g_a = Eigen::Matrix<double, 6, 1>::Zero();
        Eigen::Matrix<double, 6, 1> g_b = Eigen::Matrix<double, 6, 1>::Zero();

        for (int r = 0; r < A.rows; ++r) {
            for (int c = 0; c < nn; ++c) {
                int idx2 = A.ind_row2col[r * nn + c];
                if (idx2 == -1) break;
                double w = A.mat[r * nn + c];
                if (w <= 0) continue;

                const Eigen::Vector3d T1p = detail::transform_point(*frames[a], pc1[r]);
                const Eigen::Vector3d T2p = detail::transform_point(*frames[b], pc2[idx2]);
                const Eigen::Vector3d r_vec = T1p - T2p;
                const Eigen::Matrix<double, 3, 6> J1 = detail::right_perturbation_jacobian(*frames[a], pc1[r]);
                const Eigen::Matrix<double, 3, 6> J2 = -detail::right_perturbation_jacobian(*frames[b], pc2[idx2]);

                double sqrt_w = std::sqrt(w);
                const Eigen::Matrix<double, 3, 6> J1_w = sqrt_w * J1;
                const Eigen::Matrix<double, 3, 6> J2_w = sqrt_w * J2;
                const Eigen::Vector3d r_w = sqrt_w * r_vec;

                H_aa += J1_w.transpose() * J1_w;
                H_bb += J2_w.transpose() * J2_w;
                H_ab += J1_w.transpose() * J2_w;
                g_a  += J1_w.transpose() * r_w;
                g_b  += J2_w.transpose() * r_w;
            }
        }

        #pragma omp critical
        {
            H.block(a * pose_dof, a * pose_dof, pose_dof, pose_dof) += H_aa;
            H.block(b * pose_dof, b * pose_dof, pose_dof, pose_dof) += H_bb;
            H.block(a * pose_dof, b * pose_dof, pose_dof, pose_dof) += H_ab;
            H.block(b * pose_dof, a * pose_dof, pose_dof, pose_dof) += H_ab.transpose();
            g.segment(a * pose_dof, pose_dof) += g_a;
            g.segment(b * pose_dof, pose_dof) += g_b;
        }
    }
}

template <typename PointT>
int CvoGPU<PointT>::align_multiframe(
    const std::vector<CvoPointCloud<PointT>>& clouds,
    const pgo::MapOfPoses& initial_poses,
    const pgo::VectorOfConstraints& constraints,
    pgo::MapOfPoses* optimized_poses,
    double* registration_seconds) {

    std::map<int, size_t> id_to_idx;
    size_t idx = 0;
    for (const auto& kv : initial_poses) id_to_idx[kv.first] = idx++;
    if (clouds.size() != id_to_idx.size()) {
        std::cerr << "Number of clouds does not match number of poses\n";
        return -1;
    }

    std::vector<std::shared_ptr<CvoFrameGPU<PointT>>> frames(clouds.size());
    idx = 0;
    for (const auto& kv : initial_poses) {
        int g2o_id = kv.first;
        const pgo::Pose3d& pose = kv.second;
        Eigen::Matrix4d pose_eigen = pgo::pose3d_to_eigen<double, Eigen::RowMajor>(pose);
        double pose_arr[12];
        Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> pose_map(pose_arr);
        pose_map = pose_eigen.block<3,4>(0,0);
        frames[idx] = std::make_shared<CvoFrameGPU<PointT>>(&clouds[idx], pose_arr, params_.is_using_kdtree);
        frames[idx]->set_id(g2o_id);
        ++idx;
    }

    std::vector<std::shared_ptr<BinaryStateGPU<PointT>>> edge_states;
    for (const auto& con : constraints) {
        size_t i = id_to_idx.at(con.id_begin);
        size_t j = id_to_idx.at(con.id_end);
        edge_states.push_back(std::make_shared<BinaryStateGPU<PointT>>(
            frames[i], frames[j], &params_, params_gpu_,
            params_.multiframe_num_neighbors, params_.multiframe_ell_init));
    }

    std::vector<bool> fixed_flags(frames.size(), false);
    fixed_flags[0] = true;

    int ret = align_multiframe(frames, edge_states, fixed_flags, registration_seconds);
    if (ret != 0) return ret;

    if (optimized_poses) {
        for (size_t i = 0; i < frames.size(); ++i) {
            int g2o_id = frames[i]->get_id();
            Eigen::Map<const Eigen::Matrix<double,3,4,Eigen::RowMajor>> pose_map(frames[i]->pose_vec);
            Eigen::Matrix4d pose_eigen = Eigen::Matrix4d::Identity();
            pose_eigen.block<3,4>(0,0) = pose_map;
            (*optimized_poses)[g2o_id] = pgo::pose3d_from_eigen(pose_eigen);
        }
    }
    return 0;
}

  

} // namespace cvo
