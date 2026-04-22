#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "cvo/CvoGPU.cuh"
#include "cvo/PoseGraphOptimization.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/PointCloudLoader.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using CvoCloud = cvo::CvoPointCloud<PointType>;

namespace {

constexpr float kVoxelLeafSize = 0.01f;
constexpr std::size_t kMaxTestPoints = 128;

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

std::vector<Eigen::Matrix4f> generateGroundTruthPoses() {
    std::vector<Eigen::Matrix4f> poses(2, Eigen::Matrix4f::Identity());
    return poses;
}

std::vector<CvoCloud> makeObservedClouds(const pcl::PointCloud<pcl::PointXYZ>& base_cloud,
                                         const std::vector<Eigen::Matrix4f>& gt_poses) {
    std::vector<CvoCloud> clouds;
    clouds.reserve(gt_poses.size());
    for (std::size_t i = 0; i < gt_poses.size(); ++i) {
        clouds.emplace_back(base_cloud);
    }
    return clouds;
}

cvo::pgo::VectorOfConstraints makeConstraint(const std::vector<Eigen::Matrix4f>& gt_poses) {
    cvo::pgo::VectorOfConstraints constraints;
    cvo::pgo::Constraint3d con;
    con.id_begin = 0;
    con.id_end = 1;
    const Eigen::Matrix4f T_01 = gt_poses[0].inverse() * gt_poses[1];
    con.t_be = cvo::pgo::pose3d_from_eigen(T_01.cast<double>());
    con.information = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
    constraints.push_back(con);
    return constraints;
}

cvo::pgo::MapOfPoses makeIdentityPoses() {
    cvo::pgo::MapOfPoses poses;
    poses[0] = cvo::pgo::pose3d_from_eigen(Eigen::Matrix4d::Identity());
    poses[1] = cvo::pgo::pose3d_from_eigen(Eigen::Matrix4d::Identity());
    return poses;
}

double poseDistance(const Eigen::Matrix4d& estimate, const Eigen::Matrix4d& gt) {
    const Eigen::Matrix4d err = estimate.inverse() * gt;
    const Eigen::Matrix3d R_err = err.block<3,3>(0,0);
    const Eigen::Vector3d t_err = err.block<3,1>(0,3);
    return cvo::dist_se3<double>(R_err, t_err);
}

PointType transformPoint(const PointType& point, const Eigen::Matrix4f& transform) {
    Eigen::Vector4f ph(point.x, point.y, point.z, 1.0f);
    const Eigen::Vector3f aligned = (transform * ph).head<3>();
    PointType out = point;
    out.x = aligned.x();
    out.y = aligned.y();
    out.z = aligned.z();
    return out;
}

CvoCloud stackClouds(const std::vector<CvoCloud>& clouds,
                     const cvo::pgo::MapOfPoses& poses) {
    CvoCloud stacked;
    for (int i = 0; i < static_cast<int>(clouds.size()); ++i) {
        const Eigen::Matrix4d pose =
            cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(poses.at(i));
        const Eigen::Matrix4f inv_pose = pose.inverse().cast<float>();
        for (const auto& point : clouds[i].points()) {
            stacked.push_back(transformPoint(point, inv_pose));
        }
    }
    return stacked;
}

void writeStackedPair(const std::filesystem::path& output_path,
                      const std::vector<CvoCloud>& clouds,
                      const cvo::pgo::MapOfPoses& poses) {
    CvoCloud stacked = stackClouds(clouds, poses);
    const int ret = cvo::point_cloud_io::save_point_cloud(output_path, stacked);
    if (ret != 0) {
        throw std::runtime_error("Failed to write stacked cloud: " + output_path.string());
    }
    std::cout << "Wrote " << output_path << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <params.yaml>" << std::endl;
        return 1;
    }

    const std::filesystem::path yaml_path = std::filesystem::path(argv[1]).is_absolute()
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / argv[1];
    if (!std::filesystem::exists(yaml_path)) {
        std::cerr << "YAML file not found: " << yaml_path << std::endl;
        return 1;
    }

    const pcl::PointCloud<pcl::PointXYZ> base_cloud = loadBaseCloud();
    const std::vector<Eigen::Matrix4f> gt_poses = generateGroundTruthPoses();
    const std::vector<CvoCloud> clouds = makeObservedClouds(base_cloud, gt_poses);
    const cvo::pgo::MapOfPoses initial_poses = makeIdentityPoses();
    const cvo::pgo::VectorOfConstraints constraints = makeConstraint(gt_poses);
    const std::string stem = yaml_path.stem().string();

    const Eigen::Matrix4d gt_pose = gt_poses[1].cast<double>();
    const Eigen::Matrix4d init_pose =
        cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(initial_poses.at(1));
    const double before = poseDistance(init_pose, gt_pose);
    writeStackedPair(std::filesystem::current_path() / (stem + "_init_pair.ply"),
                     clouds, initial_poses);

    cvo::CvoParams params;
    cvo::read_CvoParams_yaml(yaml_path.c_str(), &params);
    cvo::CvoGPU<PointType> solver(params);
    cvo::pgo::MapOfPoses optimized;
    const int ret = solver.align_multiframe(clouds, initial_poses, constraints, &optimized);
    assert(ret == 0);
    writeStackedPair(std::filesystem::current_path() / (stem + "_final_pair.ply"),
                     clouds, optimized);

    const Eigen::Matrix4d final_pose =
        cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(optimized.at(1));
    const double after = poseDistance(final_pose, gt_pose);

    std::cout << "Config: " << yaml_path
              << "\nGT error before: " << before
              << "\nGT error after: " << after << std::endl;
    return 0;
}
