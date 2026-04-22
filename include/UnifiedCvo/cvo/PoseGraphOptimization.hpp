#pragma once

#if defined(CVO_USE_FULL_PGO) && __has_include(<ceres/ceres.h>) && \
    __has_include("graph_optimizer/PoseGraphOptimization.hpp") && \
    __has_include("utils/def_assert.hpp") && __has_include("utils/PoseLoader.hpp") && \
    __has_include(<sophus/so3.hpp>)
#include "graph_optimizer/PoseGraphOptimization.hpp"
#else

#include <map>
#include <vector>

#include <Eigen/Dense>

namespace cvo {
namespace pgo {

struct Pose3d {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
};

struct Constraint3d {
    int id_begin = -1;
    int id_end = -1;
    Pose3d t_be;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
};

using MapOfPoses = std::map<int, Pose3d>;
using VectorOfConstraints = std::vector<Constraint3d>;

inline Pose3d pose3d_from_eigen(const Eigen::Matrix4d& T) {
    Pose3d pose;
    pose.T = T;
    return pose;
}

template <typename Scalar, int Layout = Eigen::ColMajor>
Eigen::Matrix<Scalar, 4, 4, Layout> pose3d_to_eigen(const Pose3d& pose) {
    return pose.T.template cast<Scalar>();
}

} // namespace pgo
} // namespace cvo

#endif
