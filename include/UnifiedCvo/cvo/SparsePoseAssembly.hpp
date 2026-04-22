#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "cvo/BlockSparsePoseSystem.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State_GPU.cuh"

namespace cvo {
namespace sparse_pose {

template <typename PointT>
inline std::unordered_map<const CvoFrameGPU<PointT>*, int> build_frame_index_map(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames) {
    std::unordered_map<const CvoFrameGPU<PointT>*, int> frame_to_index;
    frame_to_index.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        frame_to_index.emplace(frames[i].get(), static_cast<int>(i));
    }
    return frame_to_index;
}

inline void initialize_block_sparse_pose_system(
    int num_poses,
    const std::vector<std::pair<int, int>>& edges,
    BlockSparsePoseSystem& system) {
    system.num_poses = num_poses;
    system.diag_blocks.assign(static_cast<size_t>(num_poses), Eigen::Matrix<double, 6, 6>::Zero());
    system.rhs_blocks.assign(static_cast<size_t>(num_poses), Eigen::Matrix<double, 6, 1>::Zero());
    system.offdiag_blocks.clear();
    system.edge_lookup.clear();
    system.offdiag_blocks.reserve(edges.size());
    system.edge_lookup.reserve(edges.size() * 2);
    for (const auto& edge_pair : edges) {
        const int i = std::min(edge_pair.first, edge_pair.second);
        const int j = std::max(edge_pair.first, edge_pair.second);
        const int idx = static_cast<int>(system.offdiag_blocks.size());
        system.edge_lookup.emplace(make_block_edge_key(i, j), idx);
        system.offdiag_blocks.push_back(BlockSparsePoseEdge{i, j, Eigen::Matrix<double, 6, 6>::Zero()});
    }
}

template <typename PointT, typename TransformFn, typename JacobianFn>
void accumulate_block_sparse_pose_edge(
    const std::shared_ptr<CvoFrameGPU<PointT>>& frame_a,
    const std::shared_ptr<CvoFrameGPU<PointT>>& frame_b,
    int a,
    int b,
    const SparseKernelMat64& A,
    int num_neighbors,
    TransformFn&& transform_point,
    JacobianFn&& right_perturbation_jacobian,
    BlockSparsePoseSystem& system) {
    const auto& pc1 = frame_a->points->points();
    const auto& pc2 = frame_b->points->points();

    Eigen::Matrix<double, 6, 6> H_aa = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> H_bb = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> H_ab = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g_a = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> g_b = Eigen::Matrix<double, 6, 1>::Zero();

    for (int r = 0; r < A.rows; ++r) {
        for (int c = 0; c < num_neighbors; ++c) {
            const int idx2 = A.ind_row2col[r * num_neighbors + c];
            if (idx2 == -1) break;
            const double w = A.mat[r * num_neighbors + c];
            if (w <= 0.0) continue;

            const Eigen::Vector3d T1p = transform_point(*frame_a, pc1[r]);
            const Eigen::Vector3d T2p = transform_point(*frame_b, pc2[idx2]);
            const Eigen::Vector3d r_vec = T1p - T2p;
            const Eigen::Matrix<double, 3, 6> J1 = right_perturbation_jacobian(*frame_a, pc1[r]);
            const Eigen::Matrix<double, 3, 6> J2 = -right_perturbation_jacobian(*frame_b, pc2[idx2]);

            const double sqrt_w = std::sqrt(w);
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

    system.diag_blocks[a] += H_aa;
    system.diag_blocks[b] += H_bb;
    system.rhs_blocks[a] += g_a;
    system.rhs_blocks[b] += g_b;

    const int i = std::min(a, b);
    const int j = std::max(a, b);
    auto& edge = system.offdiag_blocks[system.edge_lookup.at(make_block_edge_key(i, j))];
    edge.Hij += (a < b) ? H_ab : H_ab.transpose();
}

inline void finalize_block_sparse_pose_system(
    const std::vector<bool>& fixed_flags,
    BlockSparsePoseSystem& system) {
    for (size_t i = 0; i < fixed_flags.size(); ++i) {
        if (!fixed_flags[i]) {
            system.diag_blocks[i].diagonal().array() += 1e-6;
            continue;
        }
        system.diag_blocks[i].setIdentity();
        system.rhs_blocks[i].setZero();
        for (auto& edge : system.offdiag_blocks) {
            if (edge.i == static_cast<int>(i) || edge.j == static_cast<int>(i)) {
                edge.Hij.setZero();
            }
        }
    }
}

template <typename PointT, typename TransformFn, typename JacobianFn>
void build_block_sparse_pose_system(
    const std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
    const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
    const std::vector<bool>& fixed_flags,
    TransformFn&& transform_point,
    JacobianFn&& right_perturbation_jacobian,
    BlockSparsePoseSystem& system) {
    const auto frame_to_index = build_frame_index_map(frames);
    std::vector<std::pair<int, int>> edges;
    edges.reserve(edge_states.size());
    for (const auto& es : edge_states) {
        const int a = frame_to_index.at(es->frame1().get());
        const int b = frame_to_index.at(es->frame2().get());
        edges.emplace_back(a, b);
    }
    initialize_block_sparse_pose_system(static_cast<int>(frames.size()), edges, system);

    for (const auto& es : edge_states) {
        const int a = frame_to_index.at(es->frame1().get());
        const int b = frame_to_index.at(es->frame2().get());
        accumulate_block_sparse_pose_edge(
            frames[a], frames[b], a, b, es->get_A_cpu(), es->get_num_neighbors(),
            transform_point, right_perturbation_jacobian, system);
    }

    finalize_block_sparse_pose_system(fixed_flags, system);
}

}  // namespace sparse_pose
}  // namespace cvo
