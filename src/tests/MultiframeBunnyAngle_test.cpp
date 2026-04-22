#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoGPU.cuh"
#include "cvo/SparseKernelMat.hpp"
#include "utils/PointCloudLoader.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using CvoCloud = cvo::CvoPointCloud<PointType>;

namespace {

constexpr float kAngleDeg = 30.0f;
constexpr float kTranslation = 0.5f;
constexpr float kNoiseSigma = 0.01f;
constexpr float kVoxelLeafSize = 0.01f;
constexpr std::size_t kMaxTestPoints = 96;
constexpr unsigned int kSeed = 17u;

std::filesystem::path source_path() {
    return std::filesystem::path(CVO_SOURCE_DIR);
}

pcl::PointCloud<pcl::PointXYZ> loadBaseCloud() {
    const std::filesystem::path bunny_path = source_path() / "demo_data" / "bunny.pcd";
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

Eigen::Matrix4f groundTruthPose() {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) =
        Eigen::AngleAxisf(kAngleDeg * static_cast<float>(M_PI) / 180.0f,
                          Eigen::Vector3f::UnitY()).toRotationMatrix();
    T.block<3,1>(0,3) = Eigen::Vector3f(kTranslation, 0.0f, 0.0f);
    return T;
}

void addGaussianNoise(pcl::PointCloud<pcl::PointXYZ>& cloud, std::mt19937& gen) {
    std::normal_distribution<float> noise_dist(0.0f, kNoiseSigma);
    for (auto& point : cloud.points) {
        point.x += noise_dist(gen);
        point.y += noise_dist(gen);
        point.z += noise_dist(gen);
    }
}

std::pair<CvoCloud, CvoCloud> makeObservedClouds(const pcl::PointCloud<pcl::PointXYZ>& base_cloud) {
    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(base_cloud, transformed, groundTruthPose().inverse());

    std::mt19937 noise_gen(kSeed + 1u);
    addGaussianNoise(transformed, noise_gen);

    return {CvoCloud(base_cloud), CvoCloud(transformed)};
}

void toPoseArray(const Eigen::Matrix4d& T, double pose_arr[12]) {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> pose_map(pose_arr);
    pose_map = T.block<3,4>(0,0);
}

double multiframeAngle(std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>>& frames,
                       std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>>& edges) {
    for (auto& frame : frames) {
        frame->transform_pointcloud();
    }

    double angle = 0.0;
    for (auto& edge : edges) {
        edge->update_inner_product();
        const cvo::SparseKernelMat64& A = edge->get_A_cpu();
        angle += cvo::A_sum(const_cast<cvo::SparseKernelMat64*>(&A), edge->get_num_neighbors());
    }
    return angle;
}

double multiframeWeightedResidual(
    const std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>>& frames,
    const std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>>& edges) {
    return cvo::detail::multiframe_weighted_residual_cost(frames, edges);
}

double poseDistance(const Eigen::Matrix4d& estimate, const Eigen::Matrix4d& gt) {
    const Eigen::Matrix4d err = estimate.inverse() * gt;
    const Eigen::Matrix3d R_err = err.block<3,3>(0,0);
    const Eigen::Vector3d t_err = err.block<3,1>(0,3);
    return cvo::dist_se3<double>(R_err, t_err);
}

}  // namespace

int main() {
    const pcl::PointCloud<pcl::PointXYZ> base_cloud = loadBaseCloud();
    const auto [cloud0, cloud1] = makeObservedClouds(base_cloud);

    cvo::CvoParams params;
    params.multiframe_max_iters = 1;
    params.multiframe_iterations_per_ell = 1;
    params.multiframe_ell_init = 1.0f;
    params.multiframe_ell_min = 1.0f;
    params.multiframe_ell_decay_rate = 1.0f;
    params.multiframe_num_neighbors = 16;
    params.multiframe_min_nonzeros = 1;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_kdtree = 0;

    cvo::CvoGPU<PointType> solver(params);

    double pose0[12];
    double pose1[12];
    toPoseArray(Eigen::Matrix4d::Identity(), pose0);
    toPoseArray(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(
        const_cast<CvoCloud*>(&cloud0), pose0, false);
    auto frame1 = std::make_shared<cvo::CvoFrameGPU<PointType>>(
        const_cast<CvoCloud*>(&cloud1), pose1, false);

    std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>> frames = {frame0, frame1};
    std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>> edges;
    edges.push_back(std::make_shared<cvo::BinaryStateGPU<PointType>>(
        frame0, frame1, &solver.params(), solver.params_gpu(),
        params.multiframe_num_neighbors, params.multiframe_ell_init));

    const std::vector<bool> fixed_flags = {true, false};

    const Eigen::Matrix4d gt_pose = groundTruthPose().cast<double>();
    const auto currentPose = [&]() {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,4>(0,0) =
            Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(frame1->pose_vec);
        return T;
    };

    const double angle_before = multiframeAngle(frames, edges);
    const double cost_before = multiframeWeightedResidual(frames, edges);
    const double gt_error_before = poseDistance(currentPose(), gt_pose);
    const int ret = solver.align_multiframe(frames, edges, fixed_flags, nullptr);
    const double angle_after = multiframeAngle(frames, edges);
    const double cost_after = multiframeWeightedResidual(frames, edges);
    const double gt_error_after = poseDistance(currentPose(), gt_pose);

    std::cout << "Bunny angle before: " << angle_before
              << "\nBunny angle after: " << angle_after
              << "\nBunny IRLS cost before: " << cost_before
              << "\nBunny IRLS cost after: " << cost_after
              << "\nBunny GT error before: " << gt_error_before
              << "\nBunny GT error after: " << gt_error_after << std::endl;

    assert(ret == 0);
    assert(std::isfinite(angle_after));
    assert(std::isfinite(cost_after));
    assert(std::isfinite(gt_error_after));
    assert(angle_after >= angle_before);
    assert(cost_after <= cost_before);
    assert(gt_error_after < gt_error_before);
    return 0;
}
