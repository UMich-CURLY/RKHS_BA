#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "cvo/SparseKernelMat.hpp"

int main() {
    cvo::SparseKernelMat64 A;
    A.rows = 3;
    A.cols = 4;

    using RowCountT = cvo::SparseKernelMat64::RowCountType;
    using CountT = cvo::SparseKernelMat64::CountType;

    std::vector<RowCountT> host_nonzeros = {
        static_cast<RowCountT>(std::numeric_limits<std::int32_t>::max()),
        static_cast<RowCountT>(17),
        static_cast<RowCountT>(23),
    };
    const CountT expected =
        static_cast<CountT>(host_nonzeros[0]) +
        static_cast<CountT>(host_nonzeros[1]) +
        static_cast<CountT>(host_nonzeros[2]);
    assert(expected > static_cast<CountT>(std::numeric_limits<std::int32_t>::max()));

    cudaMalloc(reinterpret_cast<void**>(&A.nonzeros), sizeof(RowCountT) * host_nonzeros.size());
    cudaMemcpy(A.nonzeros,
               host_nonzeros.data(),
               sizeof(RowCountT) * host_nonzeros.size(),
               cudaMemcpyHostToDevice);

    cvo::compute_nonzeros(&A);
    std::cout << "SparseKernelMat64 nonzero_sum: " << A.nonzero_sum << std::endl;
    assert(A.nonzero_sum == expected);
    assert(A.nonzero_sum > static_cast<CountT>(std::numeric_limits<std::int32_t>::max()));

    cudaFree(A.nonzeros);
    return 0;
}
