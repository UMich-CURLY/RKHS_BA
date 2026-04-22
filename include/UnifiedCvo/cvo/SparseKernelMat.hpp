#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace cvo {

struct SparseKernelMat {
  using CountType = std::uint32_t;
  using RowCountType = std::uint32_t;

  int rows = 0;
  int cols = 0;
  CountType nonzero_sum = 0;
  float* mat = nullptr;
  int* ind_row2col = nullptr;
  RowCountType* nonzeros = nullptr;
};

struct SparseKernelMat64 {
  using CountType = std::uint64_t;
  using RowCountType = std::uint32_t;

  int rows = 0;
  int cols = 0;
  CountType nonzero_sum = 0;
  float* mat = nullptr;
  int* ind_row2col = nullptr;
  RowCountType* nonzeros = nullptr;
};

template <typename SparseMatT>
inline typename SparseMatT::CountType nonzeros(SparseMatT* A_host) {
  return A_host->nonzero_sum;
}

template <typename SparseMatT>
inline void compute_nonzeros(SparseMatT* A_host) {
  using CountT = typename SparseMatT::CountType;
  using RowCountT = typename SparseMatT::RowCountType;

  A_host->nonzero_sum = CountT{0};
  std::vector<RowCountT> v(static_cast<size_t>(A_host->rows));
  cudaMemcpy(v.data(), A_host->nonzeros, sizeof(RowCountT) * static_cast<size_t>(A_host->rows), cudaMemcpyDeviceToHost);
  A_host->nonzero_sum = std::accumulate(
      v.begin(), v.end(), CountT{0},
      [](CountT acc, RowCountT value) { return acc + static_cast<CountT>(value); });
}

template <typename SparseMatT>
inline typename SparseMatT::RowCountType max_neighbors(SparseMatT* A_host) {
  using RowCountT = typename SparseMatT::RowCountType;

  std::vector<RowCountT> v(static_cast<size_t>(A_host->rows));
  cudaMemcpy(v.data(), A_host->nonzeros, sizeof(RowCountT) * static_cast<size_t>(A_host->rows), cudaMemcpyDeviceToHost);
  return *std::max_element(v.begin(), v.end());
}

template <typename SparseMatT>
inline float A_sum(SparseMatT* A_host) {
  thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
  const size_t entries = static_cast<size_t>(A_host->rows) * static_cast<size_t>(A_host->cols);
  return thrust::reduce(A_ptr, A_ptr + entries, 0.0f);
}

template <typename SparseMatT>
inline float A_sum(SparseMatT* A_host, int num_neighbors) {
  thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->mat);
  const size_t entries = static_cast<size_t>(A_host->rows) * static_cast<size_t>(num_neighbors);
  return thrust::reduce(A_ptr, A_ptr + entries, 0.0f);
}

template <typename SparseMatT>
inline void clear_SparseKernelMat(SparseMatT* A_host) {
  using RowCountT = typename SparseMatT::RowCountType;

  A_host->nonzero_sum = 0;
  const size_t entries = static_cast<size_t>(A_host->rows) * static_cast<size_t>(A_host->cols);
  cudaMemset(A_host->mat, 0, entries * sizeof(float));
  cudaMemset(A_host->ind_row2col, -1, entries * sizeof(int));
  cudaMemset(A_host->nonzeros, 0, static_cast<size_t>(A_host->rows) * sizeof(RowCountT));
}

template <typename SparseMatT>
inline void clear_SparseKernelMat(SparseMatT* A_host, int num_neighbors) {
  using RowCountT = typename SparseMatT::RowCountType;

  A_host->nonzero_sum = 0;
  const size_t entries = static_cast<size_t>(A_host->rows) * static_cast<size_t>(num_neighbors);
  cudaMemset(A_host->mat, 0, entries * sizeof(float));
  cudaMemset(A_host->ind_row2col, -1, entries * sizeof(int));
  cudaMemset(A_host->nonzeros, 0, static_cast<size_t>(A_host->rows) * sizeof(RowCountT));
}

template <typename SparseMatT>
inline SparseMatT* init_SparseKernelMat_gpu(int row, int col, SparseMatT& A_host) {
  using RowCountT = typename SparseMatT::RowCountType;

  SparseMatT* A_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&A_out), sizeof(SparseMatT));

  A_host.rows = row;
  A_host.cols = col;
  A_host.nonzero_sum = 0;
  const size_t entries = static_cast<size_t>(row) * static_cast<size_t>(col);
  cudaMalloc(reinterpret_cast<void**>(&A_host.mat), entries * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&A_host.ind_row2col), entries * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&A_host.nonzeros), static_cast<size_t>(row) * sizeof(RowCountT));

  cudaMemcpy(A_out, &A_host, sizeof(SparseMatT), cudaMemcpyHostToDevice);
  return A_out;
}

template <typename SparseMatT>
inline void delete_SparseKernelMat_gpu(SparseMatT* A_gpu, SparseMatT* A_host) {
  cudaFree(A_host->mat);
  cudaFree(A_host->ind_row2col);
  cudaFree(A_host->nonzeros);
  cudaFree(A_gpu);
}

template <typename SparseMatT>
inline int copy_internal_SparseKernelMat_gpu_to_cpu(SparseMatT* A_host, SparseMatT* A_cpu, int num_neighbors = -1) {
  using RowCountT = typename SparseMatT::RowCountType;

  if (A_host->rows != A_cpu->rows || A_host->cols != A_cpu->cols) {
    return -1;
  }

  if (num_neighbors == -1) {
    num_neighbors = A_cpu->cols;
  }

  A_cpu->nonzero_sum = A_host->nonzero_sum;
  const size_t entries = static_cast<size_t>(num_neighbors) * static_cast<size_t>(A_host->rows);
  cudaMemcpy(A_cpu->mat, A_host->mat, entries * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_cpu->ind_row2col, A_host->ind_row2col, entries * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(A_cpu->nonzeros,
             A_host->nonzeros,
             static_cast<size_t>(A_host->rows) * sizeof(RowCountT),
             cudaMemcpyDeviceToHost);
  return 0;
}

template <typename SparseMatT>
inline void clear_SparseKernelMat_cpu(SparseMatT* A_cpu, int num_neighbors) {
  using RowCountT = typename SparseMatT::RowCountType;

  A_cpu->nonzero_sum = 0;
  const size_t entries = static_cast<size_t>(A_cpu->rows) * static_cast<size_t>(num_neighbors);
  std::memset(A_cpu->mat, 0, entries * sizeof(float));
  std::memset(A_cpu->ind_row2col, -1, entries * sizeof(int));
  std::memset(A_cpu->nonzeros, 0, static_cast<size_t>(A_cpu->rows) * sizeof(RowCountT));
}

template <typename SparseMatT>
inline void init_internal_SparseKernelMat_cpu(int rows, int cols, SparseMatT* A_cpu) {
  using RowCountT = typename SparseMatT::RowCountType;

  A_cpu->rows = rows;
  A_cpu->cols = cols;
  A_cpu->nonzero_sum = 0;
  const size_t entries = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  A_cpu->mat = new float[entries]();
  A_cpu->ind_row2col = new int[entries]();
  A_cpu->nonzeros = new RowCountT[static_cast<size_t>(rows)]();
  std::memset(A_cpu->ind_row2col, -1, entries * sizeof(int));
}

template <typename SparseMatT>
inline void delete_internal_SparseKernelMat_cpu(SparseMatT* A_cpu) {
  delete[] A_cpu->mat;
  delete[] A_cpu->ind_row2col;
  delete[] A_cpu->nonzeros;
}

}  // namespace cvo
