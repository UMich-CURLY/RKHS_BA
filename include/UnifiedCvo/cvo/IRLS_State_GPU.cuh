#pragma once

#include "CvoFrameGPU.hpp"
#include "cvo/KernelWeight.hpp"
#include "cvo/SparseKernelMat.hpp"
#include "cukdtree/cukdtree.cuh"
#include <algorithm>
#include <cmath>
#include <memory>
#include <thrust/device_vector.h>
#include <vector>

namespace cvo {

namespace detail {

inline constexpr int kMultiframeSparseFillCpuMutual = 0;
inline constexpr int kMultiframeSparseFillGpuMutual = 1;

template <typename PointT>
std::vector<std::vector<int>> compute_topk_index_lists(
    const CvoParams& params,
    const std::vector<PointT>& points_a,
    const std::vector<PointT>& points_b,
    int num_neighbors,
    double ell) {
    const double sp_thres = static_cast<double>(params.sp_thres);

    struct Candidate {
        double w = 0.0;
        int idx = -1;
    };

    std::vector<std::vector<int>> topk(points_a.size());
    std::vector<Candidate> candidates;
    candidates.reserve(points_b.size());

    for (int i = 0; i < static_cast<int>(points_a.size()); ++i) {
        candidates.clear();
        const auto& pa = points_a[i];
        for (int j = 0; j < static_cast<int>(points_b.size()); ++j) {
            const auto& pb = points_b[j];
            const double w = static_cast<double>(combined_kernel_weight(params, pa, pb, static_cast<float>(ell)));
            if (w <= static_cast<double>(sp_thres)) {
                continue;
            }
            candidates.push_back({w, j});
        }

        const int keep = std::min<int>(num_neighbors, candidates.size());
        if (keep > 0) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + keep,
                candidates.end(),
                [](const Candidate& lhs, const Candidate& rhs) {
                    if (lhs.w != rhs.w) {
                        return lhs.w > rhs.w;
                    }
                    return lhs.idx < rhs.idx;
                });
            topk[i].reserve(keep);
            for (int k = 0; k < keep; ++k) {
                topk[i].push_back(candidates[k].idx);
            }
        }
    }

    return topk;
}

template <typename PointT, typename SparseMatT>
void fill_sparse_kernel_exact_topk_host(
    const CvoParams& params,
    const std::vector<PointT>& points_a,
    const std::vector<PointT>& points_b,
    int num_neighbors,
    double ell,
    SparseMatT* A_cpu) {
    clear_SparseKernelMat_cpu(A_cpu, num_neighbors);

    const auto topk_ab = compute_topk_index_lists(params, points_a, points_b, num_neighbors, ell);
    const auto topk_ba = compute_topk_index_lists(params, points_b, points_a, num_neighbors, ell);

    for (int i = 0; i < static_cast<int>(points_a.size()); ++i) {
        int write = 0;
        const auto& pa = points_a[i];
        for (int idx_b : topk_ab[i]) {
            const auto& reverse = topk_ba[idx_b];
            if (std::find(reverse.begin(), reverse.end(), i) == reverse.end()) {
                continue;
            }

            const auto& pb = points_b[idx_b];
            const double w = static_cast<double>(combined_kernel_weight(params, pa, pb, static_cast<float>(ell)));
            if (w <= 0.0) {
                continue;
            }
            A_cpu->mat[i * num_neighbors + write] = static_cast<float>(w);
            A_cpu->ind_row2col[i * num_neighbors + write] = idx_b;
            ++write;
            if (write >= num_neighbors) {
                break;
            }
        }
        A_cpu->nonzeros[i] = static_cast<typename SparseMatT::RowCountType>(write);
    }

    A_cpu->nonzero_sum = 0;
    for (int i = 0; i < A_cpu->rows; ++i) {
        A_cpu->nonzero_sum += static_cast<typename SparseMatT::CountType>(A_cpu->nonzeros[i]);
    }
}

template <typename PointT, typename SparseMatT>
__global__ void fill_directed_sparse_from_knn_indices(
    const CvoParams* params,
    const PointT* query_points,
    int num_query,
    const PointT* ref_points,
    int num_ref,
    const int* knn_indices,
    int num_neighbors,
    float ell,
    SparseMatT* A_mat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_query) {
        return;
    }

    const PointT& pa = query_points[i];

    unsigned int write = 0;
    for (int k = 0; k < num_neighbors; ++k) {
        const int idx = knn_indices[i * num_neighbors + k];
        if (idx < 0 || idx >= num_ref) {
            continue;
        }
        const PointT& pb = ref_points[idx];
        const float w = combined_kernel_weight(*params, pa, pb, ell);
        if (w <= 0.0f) {
            continue;
        }
        A_mat->mat[i * num_neighbors + write] = w;
        A_mat->ind_row2col[i * num_neighbors + write] = idx;
        ++write;
    }
    A_mat->nonzeros[i] = static_cast<typename SparseMatT::RowCountType>(write);
}

template <typename SparseMatT>
__global__ void select_mutual_sparse_pairs(
    const SparseMatT* A_ab,
    const SparseMatT* A_ba,
    SparseMatT* A_selected,
    int num_neighbors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_ab->rows) {
        return;
    }

    const typename SparseMatT::RowCountType ab_count = A_ab->nonzeros[i];
    unsigned int write = 0;
    for (unsigned int c = 0; c < ab_count && write < static_cast<unsigned int>(num_neighbors); ++c) {
        const int idx_b = A_ab->ind_row2col[i * num_neighbors + c];
        if (idx_b < 0 || idx_b >= A_ba->rows) {
            continue;
        }
        const typename SparseMatT::RowCountType ba_count = A_ba->nonzeros[idx_b];
        bool is_mutual = false;
        for (unsigned int d = 0; d < ba_count; ++d) {
            if (A_ba->ind_row2col[idx_b * num_neighbors + d] == i) {
                is_mutual = true;
                break;
            }
        }
        if (!is_mutual) {
            continue;
        }
        A_selected->mat[i * num_neighbors + write] = A_ab->mat[i * num_neighbors + c];
        A_selected->ind_row2col[i * num_neighbors + write] = idx_b;
        ++write;
    }
    A_selected->nonzeros[i] = static_cast<typename SparseMatT::RowCountType>(write);
}

}  // namespace detail

template <typename PointT>
class BinaryStateGPU {
public:
    using Ptr = std::shared_ptr<BinaryStateGPU<PointT>>;
    using FramePtr = std::shared_ptr<CvoFrameGPU<PointT>>;

    BinaryStateGPU(FramePtr frame1,
                   FramePtr frame2,
                   const CvoParams* params_cpu,
                   const CvoParams* params_gpu,
                   unsigned int num_neighbors,
                   float init_ell);

    std::uint64_t update_inner_product();

    const SparseKernelMat64& get_A_cpu() const { return A_result_cpu_; }
    const SparseKernelMat64* get_A_gpu() const { return A_device_; }
    const SparseKernelMat64& get_A_gpu_host() const { return A_host_; }
    int get_num_neighbors() const { return num_neighbors_; }

    double get_ell() const { return ell_; }
    void set_ell(double ell) { ell_ = ell; }
    void update_ell();

    FramePtr frame1() const { return frame1_; }
    FramePtr frame2() const { return frame2_; }

    void malloc_state_memory();
    void free_state_memory();
    void free_gpu_state_memory();

private:
    FramePtr frame1_;
    FramePtr frame2_;
    const CvoParams* params_cpu_;
    const CvoParams* params_gpu_;
    int num_neighbors_;
    double ell_;

    SparseKernelMat64 A_host_;
    SparseKernelMat64* A_device_;
    SparseKernelMat64 A_result_cpu_;
    SparseKernelMat64 A_ab_host_;
    SparseKernelMat64* A_ab_device_ = nullptr;
    SparseKernelMat64 A_ba_host_;
    SparseKernelMat64* A_ba_device_ = nullptr;
    thrust::device_vector<int> nn_ab_indices_;
    thrust::device_vector<int> nn_ba_indices_;
    thrust::device_vector<PointT> tree_points_ab_;
    thrust::device_vector<PointT> tree_points_ba_;
    std::unique_ptr<perl_registration::cuKdTree<PointT>> kdtree_ab_;
    std::unique_ptr<perl_registration::cuKdTree<PointT>> kdtree_ba_;

    enum State { FREE, CPU_READY, ALLOCATED } state_ = FREE;
};

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

template <typename PointT>
BinaryStateGPU<PointT>::BinaryStateGPU(
    FramePtr frame1,
    FramePtr frame2,
    const CvoParams* params_cpu,
    const CvoParams* params_gpu,
    unsigned int num_neighbors,
    float init_ell)
    : frame1_(frame1), frame2_(frame2),
      params_cpu_(params_cpu), params_gpu_(params_gpu),
      num_neighbors_(num_neighbors), ell_(init_ell) {}

template <typename PointT>
void BinaryStateGPU<PointT>::malloc_state_memory() {
    if (state_ == FREE) {
        init_internal_SparseKernelMat_cpu(frame1_->size(), num_neighbors_, &A_result_cpu_);
        state_ = CPU_READY;
    }
    if (state_ != ALLOCATED) {
        A_device_ = init_SparseKernelMat_gpu(frame1_->size(), num_neighbors_, A_host_);
        A_ab_device_ = init_SparseKernelMat_gpu(frame1_->size(), num_neighbors_, A_ab_host_);
        A_ba_device_ = init_SparseKernelMat_gpu(frame2_->size(), num_neighbors_, A_ba_host_);
        clear_SparseKernelMat(&A_host_, num_neighbors_);
        clear_SparseKernelMat(&A_ab_host_, num_neighbors_);
        clear_SparseKernelMat(&A_ba_host_, num_neighbors_);
        nn_ab_indices_.resize(frame1_->size() * num_neighbors_);
        nn_ba_indices_.resize(frame2_->size() * num_neighbors_);
        tree_points_ab_.resize(frame2_->size());
        tree_points_ba_.resize(frame1_->size());
        kdtree_ab_ = std::make_unique<perl_registration::cuKdTree<PointT>>();
        kdtree_ba_ = std::make_unique<perl_registration::cuKdTree<PointT>>();
        state_ = ALLOCATED;
    }
}

template <typename PointT>
void BinaryStateGPU<PointT>::free_state_memory() {
    if (state_ != FREE) {
        if (state_ == ALLOCATED) {
            free_gpu_state_memory();
        }
        delete_internal_SparseKernelMat_cpu(&A_result_cpu_);
        A_result_cpu_ = SparseKernelMat64{};
        state_ = FREE;
    }
}

template <typename PointT>
void BinaryStateGPU<PointT>::free_gpu_state_memory() {
    if (state_ == ALLOCATED) {
        delete_SparseKernelMat_gpu(A_device_, &A_host_);
        delete_SparseKernelMat_gpu(A_ab_device_, &A_ab_host_);
        delete_SparseKernelMat_gpu(A_ba_device_, &A_ba_host_);
        A_device_ = nullptr;
        A_ab_device_ = nullptr;
        A_ba_device_ = nullptr;
        A_host_ = SparseKernelMat64{};
        A_ab_host_ = SparseKernelMat64{};
        A_ba_host_ = SparseKernelMat64{};
        nn_ab_indices_.clear();
        nn_ab_indices_.shrink_to_fit();
        nn_ba_indices_.clear();
        nn_ba_indices_.shrink_to_fit();
        tree_points_ab_.clear();
        tree_points_ab_.shrink_to_fit();
        tree_points_ba_.clear();
        tree_points_ba_.shrink_to_fit();
        kdtree_ab_.reset();
        kdtree_ba_.reset();
        state_ = CPU_READY;
    }
}

template <typename PointT>
std::uint64_t BinaryStateGPU<PointT>::update_inner_product() {
    malloc_state_memory();
    clear_SparseKernelMat(&A_host_, num_neighbors_);
    clear_SparseKernelMat(&A_ab_host_, num_neighbors_);
    clear_SparseKernelMat(&A_ba_host_, num_neighbors_);

    const bool use_gpu_mutual =
        params_cpu_->multiframe_sparse_fill_backend == detail::kMultiframeSparseFillGpuMutual &&
        num_neighbors_ <= perl_registration::KDTREE_K_SIZE;

    if (!use_gpu_mutual) {
        std::vector<PointT> points_a(frame1_->size());
        std::vector<PointT> points_b(frame2_->size());
        thrust::copy(frame1_->points_transformed_gpu()->points.begin(),
                     frame1_->points_transformed_gpu()->points.end(),
                     points_a.begin());
        thrust::copy(frame2_->points_transformed_gpu()->points.begin(),
                     frame2_->points_transformed_gpu()->points.end(),
                     points_b.begin());

        detail::fill_sparse_kernel_exact_topk_host(
            *params_cpu_, points_a, points_b, num_neighbors_, ell_, &A_result_cpu_);

        cudaMemcpy(A_host_.mat,
                   A_result_cpu_.mat,
                   sizeof(float) * frame1_->size() * num_neighbors_,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(A_host_.ind_row2col,
                   A_result_cpu_.ind_row2col,
                   sizeof(int) * frame1_->size() * num_neighbors_,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(A_host_.nonzeros,
                   A_result_cpu_.nonzeros,
                   sizeof(unsigned int) * frame1_->size(),
                   cudaMemcpyHostToDevice);
        A_host_.nonzero_sum = A_result_cpu_.nonzero_sum;
        return A_host_.nonzero_sum;
    }

    tree_points_ab_ = frame2_->points_transformed_gpu()->points;
    tree_points_ba_ = frame1_->points_transformed_gpu()->points;
    kdtree_ab_->SetInputCloud(tree_points_ab_);
    kdtree_ba_->SetInputCloud(tree_points_ba_);
    kdtree_ab_->NearestKSearch(frame1_->points_transformed_gpu()->points, num_neighbors_, nn_ab_indices_);
    kdtree_ba_->NearestKSearch(frame2_->points_transformed_gpu()->points, num_neighbors_, nn_ba_indices_);

    const int threads = 256;
    const int blocks_ab = static_cast<int>((frame1_->size() + threads - 1) / threads);
    const int blocks_ba = static_cast<int>((frame2_->size() + threads - 1) / threads);
    detail::fill_directed_sparse_from_knn_indices<PointT, SparseKernelMat64><<<blocks_ab, threads>>>(
        params_gpu_,
        thrust::raw_pointer_cast(frame1_->points_transformed_gpu()->points.data()),
        static_cast<int>(frame1_->size()),
        thrust::raw_pointer_cast(frame2_->points_transformed_gpu()->points.data()),
        static_cast<int>(frame2_->size()),
        thrust::raw_pointer_cast(nn_ab_indices_.data()),
        num_neighbors_,
        static_cast<float>(ell_),
        A_ab_device_);
    detail::fill_directed_sparse_from_knn_indices<PointT, SparseKernelMat64><<<blocks_ba, threads>>>(
        params_gpu_,
        thrust::raw_pointer_cast(frame2_->points_transformed_gpu()->points.data()),
        static_cast<int>(frame2_->size()),
        thrust::raw_pointer_cast(frame1_->points_transformed_gpu()->points.data()),
        static_cast<int>(frame1_->size()),
        thrust::raw_pointer_cast(nn_ba_indices_.data()),
        num_neighbors_,
        static_cast<float>(ell_),
        A_ba_device_);
    detail::select_mutual_sparse_pairs<SparseKernelMat64><<<blocks_ab, threads>>>(
        A_ab_device_, A_ba_device_, A_device_, num_neighbors_);
    cudaDeviceSynchronize();

    compute_nonzeros(&A_host_);
    copy_internal_SparseKernelMat_gpu_to_cpu(&A_host_, &A_result_cpu_, num_neighbors_);
    return A_host_.nonzero_sum;
}

template <typename PointT>
void BinaryStateGPU<PointT>::update_ell() {
    ell_ = std::max(ell_ * params_cpu_->multiframe_ell_decay_rate,
                    static_cast<double>(params_cpu_->multiframe_ell_min));
}

} // namespace cvo
