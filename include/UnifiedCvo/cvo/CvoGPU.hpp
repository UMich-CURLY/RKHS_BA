#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "cvo/BlockSparsePoseSystem.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/Association.hpp"
#include "cvo/IRLS_State_GPU.cuh"
#include "cvo/PoseGraphOptimization.hpp"
#include "cvo/SparsePoseSolver.hpp"
#include "utils/CvoPointCloud.hpp"

namespace cvo {

struct CvoResultInfo {
    int return_code = 0;
    Eigen::Matrix4f T_s2t = Eigen::Matrix4f::Identity();
    Association association;
    double registration_seconds = 0.0;
    int num_iters = 0;
};

struct MultiframeDebugInfo {
    bool valid = false;
    double ell = 0.0;
    int ell_level = -1;
    int iter_in_ell = -1;
    int min_nonzeros = 0;
    double baseline_cost = 0.0;
    double baseline_angle = 0.0;
    double gradient_norm = 0.0;
    double raw_step_norm = 0.0;
    double accepted_step_norm = 0.0;
    Eigen::Matrix<double, 6, 1> gradient_frame = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian_block = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> raw_step = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> accepted_step = Eigen::Matrix<double, 6, 1>::Zero();
};

template <typename PointT>
class CvoGPU {
public:
    using PointCloud = CvoPointCloud<PointT>;

    explicit CvoGPU(const CvoParams &params);
    ~CvoGPU();

    // Two‑frame alignment only.
    // Convention: T * p_target = p_source  (i.e., transform from target frame to source frame).
    CvoResultInfo align(const PointCloud & source,
                        const PointCloud & target,
                        const Eigen::Matrix4f & T_init,   // T_init * target = source
                        bool return_association = false) const;

    // Helper: inner product / angle
    float function_angle(const PointCloud & source,
                         const PointCloud & target,
                         const Eigen::Matrix4f & T,       // T * target = source
                         float ell,
                         bool approximate = true) const;

    // (Optional) compute association matrix
    void compute_association(const PointCloud & source,
                             const PointCloud & target,
                             const Eigen::Matrix4f & T,
                             float ell,
                             Association & assoc) const;

    // Expert interface
    int align_multiframe(std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
                         const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
                         const std::vector<bool>& fixed_flags,
                         double* registration_seconds = nullptr);

    // User‑friendly interface (g2o)
    int align_multiframe(const std::vector<CvoPointCloud<PointT>>& clouds,
                         const pgo::MapOfPoses& initial_poses,
                         const pgo::VectorOfConstraints& constraints,
                         pgo::MapOfPoses* optimized_poses = nullptr,
                         double* registration_seconds = nullptr);

    // Access to parameters
    CvoParams & params() { return params_; }
    const CvoParams * params_gpu() const { return params_gpu_; }
    const MultiframeDebugInfo& last_multiframe_debug() const { return last_multiframe_debug_; }

 

private:
    void build_linear_system(const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
                             const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
                             const std::vector<bool>& fixed_flags,
                             Eigen::MatrixXd& H,
                             Eigen::VectorXd& g);

    void build_linear_system_gpu(const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
                                 const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
                                 const std::vector<bool>& fixed_flags,
                                 Eigen::MatrixXd& H,
                                 Eigen::VectorXd& g);
    void build_block_sparse_system(const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
                                   const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
                                   const std::vector<bool>& fixed_flags,
                                   BlockSparsePoseSystem& system);
    bool solve_sparse_block_system_cpu(const BlockSparsePoseSystem& system, Eigen::VectorXd& dx);
    bool solve_sparse_block_system_gpu(const BlockSparsePoseSystem& system, Eigen::VectorXd& dx);

    void ensure_multiframe_system_buffers(int system_size);
    void release_multiframe_system_buffers();
    double evaluate_multiframe_weighted_residual_cost_gpu(
        const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
        const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states);
    double evaluate_multiframe_angle_objective_gpu(
        const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states);
  
    CvoParams params_;
    CvoParams * params_gpu_;   // device copy
    MultiframeDebugInfo last_multiframe_debug_;
    double* multiframe_H_device_ = nullptr;
    double* multiframe_g_device_ = nullptr;
    double* multiframe_cost_device_ = nullptr;
    int multiframe_system_size_ = 0;
    CpuSparsePoseSolver cpu_sparse_pose_solver_;
    GpuSparsePoseSolver gpu_sparse_pose_solver_;
};

} // namespace cvo
