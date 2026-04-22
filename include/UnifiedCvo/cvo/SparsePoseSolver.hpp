#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <cublas_v2.h>
#include <cusparse.h>
#include <Eigen/Sparse>

#include "cvo/BlockSparsePoseSystem.hpp"

namespace cvo {

inline void build_block_sparse_csr_pattern(const BlockSparsePoseSystem& system,
                                           BlockSparsePoseCsrPattern& pattern) {
    pattern.n = system.num_poses * 6;
    pattern.edge_pairs.clear();
    pattern.edge_pairs.reserve(system.offdiag_blocks.size());
    for (const auto& edge : system.offdiag_blocks) {
        pattern.edge_pairs.emplace_back(edge.i, edge.j);
    }

    std::vector<std::vector<int>> cols(static_cast<size_t>(pattern.n));
    for (int p = 0; p < system.num_poses; ++p) {
        const int base = p * 6;
        for (int r = 0; r < 6; ++r) {
            auto& row = cols[static_cast<size_t>(base + r)];
            for (int c = 0; c < 6; ++c) {
                row.push_back(base + c);
            }
        }
    }
    for (const auto& edge : system.offdiag_blocks) {
        const int bi = edge.i * 6;
        const int bj = edge.j * 6;
        for (int r = 0; r < 6; ++r) {
            auto& row_i = cols[static_cast<size_t>(bi + r)];
            auto& row_j = cols[static_cast<size_t>(bj + r)];
            for (int c = 0; c < 6; ++c) {
                row_i.push_back(bj + c);
                row_j.push_back(bi + c);
            }
        }
    }

    pattern.row_offsets.assign(static_cast<size_t>(pattern.n) + 1, 0);
    pattern.col_indices.clear();
    for (int row = 0; row < pattern.n; ++row) {
        auto& row_cols = cols[static_cast<size_t>(row)];
        std::sort(row_cols.begin(), row_cols.end());
        row_cols.erase(std::unique(row_cols.begin(), row_cols.end()), row_cols.end());
        pattern.row_offsets[static_cast<size_t>(row + 1)] =
            pattern.row_offsets[static_cast<size_t>(row)] + static_cast<int>(row_cols.size());
        pattern.col_indices.insert(pattern.col_indices.end(), row_cols.begin(), row_cols.end());
    }
    pattern.nnz = static_cast<int>(pattern.col_indices.size());
    pattern.values.assign(static_cast<size_t>(pattern.nnz), 0.0);
    pattern.rhs.assign(static_cast<size_t>(pattern.n), 0.0);
}

inline void fill_block_sparse_csr_values(const BlockSparsePoseSystem& system,
                                         BlockSparsePoseCsrPattern& pattern) {
    std::fill(pattern.values.begin(), pattern.values.end(), 0.0);
    std::fill(pattern.rhs.begin(), pattern.rhs.end(), 0.0);

    auto add_value = [&](int row, int col, double value) {
        const int begin = pattern.row_offsets[static_cast<size_t>(row)];
        const int end = pattern.row_offsets[static_cast<size_t>(row + 1)];
        auto it = std::lower_bound(pattern.col_indices.begin() + begin, pattern.col_indices.begin() + end, col);
        if (it != pattern.col_indices.begin() + end && *it == col) {
            pattern.values[static_cast<size_t>(it - pattern.col_indices.begin())] += value;
        }
    };

    for (int p = 0; p < system.num_poses; ++p) {
        const int base = p * 6;
        for (int r = 0; r < 6; ++r) {
            pattern.rhs[static_cast<size_t>(base + r)] = -system.rhs_blocks[p](r);
            for (int c = 0; c < 6; ++c) {
                add_value(base + r, base + c, system.diag_blocks[p](r, c));
            }
        }
    }
    for (const auto& edge : system.offdiag_blocks) {
        const int bi = edge.i * 6;
        const int bj = edge.j * 6;
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                add_value(bi + r, bj + c, edge.Hij(r, c));
                add_value(bj + r, bi + c, edge.Hij(c, r));
            }
        }
    }
}

class CpuSparsePoseSolver {
public:
    bool solve(const BlockSparsePoseSystem& system, Eigen::VectorXd& dx) {
        if (!pattern_.matches(system)) {
            build_block_sparse_csr_pattern(system, pattern_);
        }
        fill_block_sparse_csr_values(system, pattern_);
        Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor, int>> H(
            pattern_.n, pattern_.n, pattern_.nnz,
            pattern_.row_offsets.data(), pattern_.col_indices.data(), pattern_.values.data());
        Eigen::Map<const Eigen::VectorXd> b(pattern_.rhs.data(), pattern_.n);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor, int>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) return false;
        dx = solver.solve(b);
        return dx.allFinite();
    }

private:
    BlockSparsePoseCsrPattern pattern_;
};

class GpuSparsePoseSolver {
public:
    ~GpuSparsePoseSolver() { release(); }

    bool solve(const BlockSparsePoseSystem& system, Eigen::VectorXd& dx) {
        if (!pattern_.matches(system)) {
            build_block_sparse_csr_pattern(system, pattern_);
            allocate_pattern();
        }
        fill_block_sparse_csr_values(system, pattern_);
        return run_pcg(dx);
    }

private:
    void allocate_pattern() {
        release();
        cudaMalloc(reinterpret_cast<void**>(&d_row_offsets_), sizeof(int) * pattern_.row_offsets.size());
        cudaMalloc(reinterpret_cast<void**>(&d_col_indices_), sizeof(int) * pattern_.col_indices.size());
        cudaMalloc(reinterpret_cast<void**>(&d_values_), sizeof(double) * pattern_.values.size());
        cudaMalloc(reinterpret_cast<void**>(&d_x_), sizeof(double) * static_cast<size_t>(pattern_.n));
        cudaMalloc(reinterpret_cast<void**>(&d_r_), sizeof(double) * static_cast<size_t>(pattern_.n));
        cudaMalloc(reinterpret_cast<void**>(&d_p_), sizeof(double) * static_cast<size_t>(pattern_.n));
        cudaMalloc(reinterpret_cast<void**>(&d_Ap_), sizeof(double) * static_cast<size_t>(pattern_.n));

        cudaMemcpy(d_row_offsets_, pattern_.row_offsets.data(),
                   sizeof(int) * pattern_.row_offsets.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices_, pattern_.col_indices.data(),
                   sizeof(int) * pattern_.col_indices.size(), cudaMemcpyHostToDevice);

        cublasCreate(&cublas_);
        cusparseCreate(&cusparse_);
        cusparseCreateCsr(&matA_, pattern_.n, pattern_.n, pattern_.nnz,
                          d_row_offsets_, d_col_indices_, d_values_,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateDnVec(&vecP_, pattern_.n, d_p_, CUDA_R_64F);
        cusparseCreateDnVec(&vecAp_, pattern_.n, d_Ap_, CUDA_R_64F);

        size_t buffer_size = 0;
        const double alpha = 1.0;
        const double beta = 0.0;
        cusparseSpMV_bufferSize(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA_, vecP_, &beta, vecAp_, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
        cudaMalloc(&spmv_buffer_, buffer_size);
    }

    bool run_pcg(Eigen::VectorXd& dx) {
        cudaMemcpy(d_values_, pattern_.values.data(),
                   sizeof(double) * pattern_.values.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_, pattern_.rhs.data(),
                   sizeof(double) * pattern_.rhs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p_, pattern_.rhs.data(),
                   sizeof(double) * pattern_.rhs.size(), cudaMemcpyHostToDevice);
        cudaMemset(d_x_, 0, sizeof(double) * static_cast<size_t>(pattern_.n));

        double rsold = 0.0;
        cublasDdot(cublas_, pattern_.n, d_r_, 1, d_r_, 1, &rsold);
        const double tol2 = 1e-12;
        bool ok = false;
        const double alpha = 1.0;
        const double zero = 0.0;
        for (int iter = 0; iter < std::min(4 * pattern_.n, 2000); ++iter) {
            cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA_, vecP_, &zero, vecAp_, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_);
            double pAp = 0.0;
            cublasDdot(cublas_, pattern_.n, d_p_, 1, d_Ap_, 1, &pAp);
            if (std::abs(pAp) < 1e-30) break;
            const double alpha_k = rsold / pAp;
            cublasDaxpy(cublas_, pattern_.n, &alpha_k, d_p_, 1, d_x_, 1);
            const double neg_alpha_k = -alpha_k;
            cublasDaxpy(cublas_, pattern_.n, &neg_alpha_k, d_Ap_, 1, d_r_, 1);
            double rsnew = 0.0;
            cublasDdot(cublas_, pattern_.n, d_r_, 1, d_r_, 1, &rsnew);
            if (rsnew < tol2) {
                ok = true;
                break;
            }
            const double beta_k = rsnew / rsold;
            cublasDscal(cublas_, pattern_.n, &beta_k, d_p_, 1);
            const double one = 1.0;
            cublasDaxpy(cublas_, pattern_.n, &one, d_r_, 1, d_p_, 1);
            rsold = rsnew;
        }
        dx.resize(pattern_.n);
        cudaMemcpy(dx.data(), d_x_, sizeof(double) * static_cast<size_t>(pattern_.n), cudaMemcpyDeviceToHost);
        return ok && dx.allFinite();
    }

    void release() {
        if (vecP_) cusparseDestroyDnVec(vecP_);
        if (vecAp_) cusparseDestroyDnVec(vecAp_);
        if (matA_) cusparseDestroySpMat(matA_);
        if (cusparse_) cusparseDestroy(cusparse_);
        if (cublas_) cublasDestroy(cublas_);
        if (spmv_buffer_) cudaFree(spmv_buffer_);
        if (d_row_offsets_) cudaFree(d_row_offsets_);
        if (d_col_indices_) cudaFree(d_col_indices_);
        if (d_values_) cudaFree(d_values_);
        if (d_x_) cudaFree(d_x_);
        if (d_r_) cudaFree(d_r_);
        if (d_p_) cudaFree(d_p_);
        if (d_Ap_) cudaFree(d_Ap_);
        vecP_ = nullptr;
        vecAp_ = nullptr;
        matA_ = nullptr;
        cusparse_ = nullptr;
        cublas_ = nullptr;
        spmv_buffer_ = nullptr;
        d_row_offsets_ = nullptr;
        d_col_indices_ = nullptr;
        d_values_ = nullptr;
        d_x_ = nullptr;
        d_r_ = nullptr;
        d_p_ = nullptr;
        d_Ap_ = nullptr;
    }

    BlockSparsePoseCsrPattern pattern_;
    int* d_row_offsets_ = nullptr;
    int* d_col_indices_ = nullptr;
    double* d_values_ = nullptr;
    double* d_x_ = nullptr;
    double* d_r_ = nullptr;
    double* d_p_ = nullptr;
    double* d_Ap_ = nullptr;
    void* spmv_buffer_ = nullptr;
    cusparseHandle_t cusparse_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cusparseSpMatDescr_t matA_ = nullptr;
    cusparseDnVecDescr_t vecP_ = nullptr;
    cusparseDnVecDescr_t vecAp_ = nullptr;
};

}  // namespace cvo
