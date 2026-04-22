#pragma once

#include <algorithm>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cvo/Association.hpp"
#include "cvo/SparseKernelMat.hpp"

namespace cvo {

inline void gpu_association_to_cpu(const SparseKernelMat& association_gpu,
                                   Association& association_cpu,
                                   int num_source,
                                   int num_target,
                                   int num_neighbors = -1) {
    const int rows = association_gpu.rows;
    const int cols = num_neighbors == -1 ? association_gpu.cols : num_neighbors;
    (void)num_source;
    (void)num_target;

    association_cpu.source_inliers.clear();
    association_cpu.target_inliers.clear();
    association_cpu.pairs.resize(rows, association_gpu.cols);

    if (association_gpu.nonzero_sum == 0 || rows == 0 || cols == 0) {
        association_cpu.pairs.setZero();
        association_cpu.pairs.makeCompressed();
        return;
    }

    thrust::device_ptr<float> inner_product_ptr(thrust::raw_pointer_cast(association_gpu.mat));
    thrust::host_vector<float> inner_product(inner_product_ptr, inner_product_ptr + rows * cols);

    thrust::device_ptr<int> ind_row2col_ptr(thrust::raw_pointer_cast(association_gpu.ind_row2col));
    thrust::host_vector<int> ind_row2col(ind_row2col_ptr, ind_row2col_ptr + rows * cols);

    thrust::device_ptr<unsigned int> nonzeros_ptr(thrust::raw_pointer_cast(association_gpu.nonzeros));
    thrust::host_vector<unsigned int> nonzeros(nonzeros_ptr, nonzeros_ptr + rows);

    std::vector<char> target_seen(static_cast<size_t>(association_gpu.cols), 0);
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(association_gpu.nonzero_sum);

    for (int i = 0; i < rows; ++i) {
        if (nonzeros[i] == 0) {
            continue;
        }
        association_cpu.source_inliers.push_back(i);

        for (int j = 0; j < cols; ++j) {
            const int target_index = ind_row2col[i * cols + j];
            if (target_index < 0 || target_index >= association_gpu.cols) {
                break;
            }
            if (!target_seen[static_cast<size_t>(target_index)]) {
                association_cpu.target_inliers.push_back(target_index);
                target_seen[static_cast<size_t>(target_index)] = 1;
            }
            triplets.emplace_back(i, target_index, inner_product[i * cols + j]);
        }
    }

    std::sort(association_cpu.target_inliers.begin(), association_cpu.target_inliers.end());
    association_cpu.target_inliers.erase(
        std::unique(association_cpu.target_inliers.begin(), association_cpu.target_inliers.end()),
        association_cpu.target_inliers.end());
    association_cpu.pairs.setFromTriplets(triplets.begin(), triplets.end());
    association_cpu.pairs.makeCompressed();
}

} // namespace cvo
