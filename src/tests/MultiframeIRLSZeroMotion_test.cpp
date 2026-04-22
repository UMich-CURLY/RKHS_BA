#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "cvo/CvoGPU.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/PointCloudLoader.hpp"
#include "utils/PointSemantic.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using PointCloud = cvo::CvoPointCloud<PointType>;

namespace {

constexpr float kVoxelLeafSize = 0.01f;
constexpr std::size_t kMaxTestPoints = 128;

std::filesystem::path sourcePath() {
    return std::filesystem::path(CVO_SOURCE_DIR);
}

pcl::PointCloud<pcl::PointXYZ> loadBaseCloud() {
    const std::filesystem::path bunny_path = sourcePath() / "demo_data" / "bunny.pcd";
    pcl::PointCloud<pcl::PointXYZ> raw;
    const int ret = cvo::point_cloud_io::load_point_cloud(bunny_path, raw);
    if (ret != 0) {
        throw std::runtime_error("Failed to load bunny point cloud: " + bunny_path.string());
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(raw.makeShared());
    voxel.setLeafSize(kVoxelLeafSize, kVoxelLeafSize, kVoxelLeafSize);

    pcl::PointCloud<pcl::PointXYZ> downsampled;
    voxel.filter(downsampled);
    if (downsampled.size() <= kMaxTestPoints) {
        return downsampled;
    }

    pcl::PointCloud<pcl::PointXYZ> reduced;
    reduced.reserve(kMaxTestPoints);
    const double stride = static_cast<double>(downsampled.size() - 1) /
                          static_cast<double>(kMaxTestPoints - 1);
    for (std::size_t i = 0; i < kMaxTestPoints; ++i) {
        const std::size_t idx = static_cast<std::size_t>(std::llround(i * stride));
        reduced.push_back(downsampled[idx]);
    }
    reduced.width = static_cast<std::uint32_t>(reduced.size());
    reduced.height = 1;
    reduced.is_dense = false;
    return reduced;
}

void toPoseArray(const Eigen::Matrix4d& T, double pose_arr[12]) {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> pose_map(pose_arr);
    pose_map = T.block<3,4>(0,0);
}

double poseDistanceFromIdentity(const double pose_arr[12]) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,4>(0,0) =
        Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(pose_arr);
    const Eigen::Matrix3d R = T.block<3,3>(0,0);
    const Eigen::Vector3d t = T.block<3,1>(0,3);
    return cvo::dist_se3<double>(R, t);
}

}  // namespace

void run_zero_motion_case(const PointCloud& cloud,
                          int num_neighbors,
                          int sparse_backend,
                          int linear_backend,
                          int objective_backend,
                          int enable_line_search,
                          const char* label) {
    cvo::CvoParams params;
    params.multiframe_max_iters = 1;
    params.multiframe_iterations_per_ell = 1;
    params.multiframe_ell_init = 1.0f;
    params.multiframe_ell_min = 1.0f;
    params.multiframe_ell_decay_rate = 1.0f;
    params.multiframe_num_neighbors = num_neighbors;
    params.multiframe_min_nonzeros = 1;
    params.multiframe_sparse_fill_backend = sparse_backend;
    params.multiframe_linear_system_backend = linear_backend;
    params.multiframe_objective_eval_backend = objective_backend;
    params.multiframe_enable_line_search = enable_line_search;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_kdtree = 0;

    cvo::CvoGPU<PointType> solver(params);

    double pose0[12];
    double pose1[12];
    toPoseArray(Eigen::Matrix4d::Identity(), pose0);
    toPoseArray(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(
        const_cast<PointCloud*>(&cloud), pose0, false);
    auto frame1 = std::make_shared<cvo::CvoFrameGPU<PointType>>(
        const_cast<PointCloud*>(&cloud), pose1, false);

    std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>> frames = {frame0, frame1};
    std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>> edges;
    edges.push_back(std::make_shared<cvo::BinaryStateGPU<PointType>>(
        frame0, frame1, &solver.params(), solver.params_gpu(),
        params.multiframe_num_neighbors, params.multiframe_ell_init));

    const std::vector<bool> fixed_flags = {true, false};

    const int ret = solver.align_multiframe(frames, edges, fixed_flags, nullptr);
    const auto& debug = solver.last_multiframe_debug();
    const double pose_err_after = poseDistanceFromIdentity(frame1->pose_vec);

    std::cout << label
              << "\nZero-motion debug valid: " << debug.valid
              << "\nGradient norm: " << debug.gradient_norm
              << "\nGradient frame: " << debug.gradient_frame.transpose()
              << "\nRaw step norm: " << debug.raw_step_norm
              << "\nRaw step: " << debug.raw_step.transpose()
              << "\nAccepted step norm: " << debug.accepted_step_norm
              << "\nAccepted step: " << debug.accepted_step.transpose()
              << "\nPose error after: " << pose_err_after << std::endl;

    assert(ret == 0);
    if (debug.valid) {
        assert(std::isfinite(debug.gradient_norm));
        assert(std::isfinite(debug.accepted_step_norm));
        assert(debug.gradient_norm < 1e-8);
        assert(debug.accepted_step_norm < 1e-8);
    }
    assert(std::isfinite(pose_err_after));
    assert(pose_err_after < 1e-8);
}

int main() {
    const PointCloud cloud(loadBaseCloud());
    run_zero_motion_case(cloud, 16, 0, 0, 0, 1, "Sparse mutual top-k CPU fill + CPU linear");
    run_zero_motion_case(cloud, 16, 1, 0, 0, 1, "Sparse mutual top-k GPU fill + CPU linear");
    run_zero_motion_case(cloud, 16, 1, 1, 0, 1, "Sparse mutual top-k GPU fill + GPU linear");
    run_zero_motion_case(cloud, 16, 1, 2, 0, 1, "Sparse mutual top-k GPU fill + CPU sparse linear");
    run_zero_motion_case(cloud, 16, 1, 3, 0, 1, "Sparse mutual top-k GPU fill + GPU sparse linear");
    run_zero_motion_case(cloud, 16, 1, 1, 1, 1, "Sparse mutual top-k GPU fill + GPU linear + GPU objective");
    run_zero_motion_case(cloud, 16, 1, 1, 1, 0, "Sparse mutual top-k GPU fill + GPU linear + GPU objective + no line search");
    run_zero_motion_case(cloud, static_cast<int>(cloud.size()), 1, 1, 1, 1, "Full neighbors reference");
    return 0;
}
