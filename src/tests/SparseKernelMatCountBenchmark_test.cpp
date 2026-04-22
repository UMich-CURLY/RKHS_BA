#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cvo/SparseKernelMat.hpp"

namespace {

template <typename SparseMatT>
double benchmark_compute_nonzeros(int rows,
                                  std::uint32_t row_value,
                                  int warmup_iters,
                                  int timed_iters,
                                  typename SparseMatT::CountType expected_sum) {
    using RowCountT = typename SparseMatT::RowCountType;

    SparseMatT A;
    A.rows = rows;
    A.cols = 1;

    std::vector<RowCountT> host_nonzeros(static_cast<size_t>(rows), static_cast<RowCountT>(row_value));
    cudaMalloc(reinterpret_cast<void**>(&A.nonzeros), sizeof(RowCountT) * host_nonzeros.size());
    cudaMemcpy(A.nonzeros,
               host_nonzeros.data(),
               sizeof(RowCountT) * host_nonzeros.size(),
               cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup_iters; ++i) {
        cvo::compute_nonzeros(&A);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_iters; ++i) {
        cvo::compute_nonzeros(&A);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    assert(A.nonzero_sum == expected_sum);
    cudaFree(A.nonzeros);

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / static_cast<double>(timed_iters);
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = (argc > 1) ? std::stoi(argv[1]) : 1 << 20;
    const int timed_iters = (argc > 2) ? std::stoi(argv[2]) : 200;
    const int warmup_iters = (argc > 3) ? std::stoi(argv[3]) : 20;
    const std::uint32_t row_value = (argc > 4) ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 1024u;

    if (rows <= 0 || timed_iters <= 0 || warmup_iters < 0) {
        std::cerr << "Usage: " << argv[0] << " [rows>0] [timed_iters>0] [warmup_iters>=0] [row_value]\n";
        return 1;
    }

    const std::uint64_t expected64 = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(row_value);
    const std::uint32_t expected32 = static_cast<std::uint32_t>(expected64 & std::numeric_limits<std::uint32_t>::max());
    const bool overflows_u32 = expected64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

    const double avg32_ms = benchmark_compute_nonzeros<cvo::SparseKernelMat>(
        rows, row_value, warmup_iters, timed_iters, expected32);
    const double avg64_ms = benchmark_compute_nonzeros<cvo::SparseKernelMat64>(
        rows, row_value, warmup_iters, timed_iters, expected64);

    std::cout << "rows=" << rows
              << " row_value=" << row_value
              << " expected_sum_u64=" << expected64
              << " overflow_u32=" << (overflows_u32 ? 1 : 0) << '\n'
              << "avg_ms_32=" << avg32_ms << '\n'
              << "avg_ms_64=" << avg64_ms << '\n'
              << "ratio_64_over_32=" << (avg64_ms / avg32_ms) << std::endl;
    return 0;
}
