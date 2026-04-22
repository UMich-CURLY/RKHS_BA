#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace cvo {

struct BlockSparsePoseEdge {
    int i = -1;
    int j = -1;
    Eigen::Matrix<double, 6, 6> Hij = Eigen::Matrix<double, 6, 6>::Zero();
};

struct BlockSparsePoseSystem {
    int num_poses = 0;
    std::vector<Eigen::Matrix<double, 6, 6>> diag_blocks;
    std::vector<Eigen::Matrix<double, 6, 1>> rhs_blocks;
    std::vector<BlockSparsePoseEdge> offdiag_blocks;
    std::unordered_map<std::uint64_t, int> edge_lookup;
};

inline std::uint64_t make_block_edge_key(int i, int j) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(i)) << 32) |
           static_cast<std::uint32_t>(j);
}

struct BlockSparsePoseCsrPattern {
    int n = 0;
    int nnz = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<std::pair<int, int>> edge_pairs;

    bool matches(const BlockSparsePoseSystem& system) const {
        if (n != system.num_poses * 6) {
            return false;
        }
        if (edge_pairs.size() != system.offdiag_blocks.size()) {
            return false;
        }
        for (size_t idx = 0; idx < edge_pairs.size(); ++idx) {
            if (edge_pairs[idx].first != system.offdiag_blocks[idx].i ||
                edge_pairs[idx].second != system.offdiag_blocks[idx].j) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace cvo
