#include <cstdint>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
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

constexpr float kMaxAngleDeg = 30.0f;
constexpr float kMaxTranslation = 0.5f;
constexpr float kNoiseSigma = 0.01f;
constexpr float kVoxelLeafSize = 0.01f;
constexpr std::size_t kMaxTestPoints = 512;
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

std::vector<Eigen::Matrix4f> generateGroundTruthPoses(int num_frames) {
    std::vector<Eigen::Matrix4f> poses(num_frames, Eigen::Matrix4f::Identity());
    if (num_frames == 2) {
        poses[1].block<3,3>(0,0) =
            Eigen::AngleAxisf(kMaxAngleDeg * static_cast<float>(M_PI) / 180.0f,
                              Eigen::Vector3f::UnitY()).toRotationMatrix();
        poses[1].block<3,1>(0,3) = Eigen::Vector3f(kMaxTranslation, 0.0f, 0.0f);
        return poses;
    }

    std::mt19937 gen(kSeed);
    std::uniform_real_distribution<float> axis_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> angle_dist(0.0f, kMaxAngleDeg * static_cast<float>(M_PI) / 180.0f);
    std::uniform_real_distribution<float> radius_dist(0.0f, kMaxTranslation);

    for (int i = 1; i < num_frames; ++i) {
        Eigen::Vector3f axis(axis_dist(gen), axis_dist(gen), axis_dist(gen));
        while (axis.norm() < 1e-4f) {
            axis = Eigen::Vector3f(axis_dist(gen), axis_dist(gen), axis_dist(gen));
        }
        axis.normalize();

        Eigen::Vector3f direction(axis_dist(gen), axis_dist(gen), axis_dist(gen));
        while (direction.norm() < 1e-4f) {
            direction = Eigen::Vector3f(axis_dist(gen), axis_dist(gen), axis_dist(gen));
        }
        direction.normalize();

        const float angle = angle_dist(gen);
        const float radius = radius_dist(gen);

        poses[i].block<3,3>(0,0) = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
        poses[i].block<3,1>(0,3) = radius * direction;
    }

    return poses;
}

void addGaussianNoise(pcl::PointCloud<pcl::PointXYZ>& cloud, std::mt19937& gen) {
    std::normal_distribution<float> noise_dist(0.0f, kNoiseSigma);
    for (auto& point : cloud.points) {
        point.x += noise_dist(gen);
        point.y += noise_dist(gen);
        point.z += noise_dist(gen);
    }
}

std::vector<CvoCloud> makeObservedClouds(const pcl::PointCloud<pcl::PointXYZ>& base_cloud,
                                         const std::vector<Eigen::Matrix4f>& gt_poses) {
    std::mt19937 noise_gen(kSeed + 1u);
    std::vector<CvoCloud> clouds;
    clouds.reserve(gt_poses.size());

    for (const Eigen::Matrix4f& pose : gt_poses) {
        pcl::PointCloud<pcl::PointXYZ> transformed;
        pcl::transformPointCloud(base_cloud, transformed, pose.inverse());
        addGaussianNoise(transformed, noise_gen);
        clouds.emplace_back(transformed);
    }
    return clouds;
}

cvo::pgo::VectorOfConstraints makeAllPairConstraints(const std::vector<Eigen::Matrix4f>& gt_poses) {
    cvo::pgo::VectorOfConstraints constraints;
    for (int j = 1; j < static_cast<int>(gt_poses.size()); ++j) {
        cvo::pgo::Constraint3d con;
        con.id_begin = 0;
        con.id_end = j;
        const Eigen::Matrix4f T_0j = gt_poses[0].inverse() * gt_poses[j];
        con.t_be = cvo::pgo::pose3d_from_eigen(T_0j.cast<double>());
        con.information = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
        constraints.push_back(con);
    }
    return constraints;
}

cvo::pgo::MapOfPoses makeIdentityPoses(int num_frames) {
    cvo::pgo::MapOfPoses poses;
    for (int i = 0; i < num_frames; ++i) {
        poses[i] = cvo::pgo::pose3d_from_eigen(Eigen::Matrix4d::Identity());
    }
    return poses;
}

double poseDistance(const Eigen::Matrix4d& estimate, const Eigen::Matrix4d& gt) {
    const Eigen::Matrix4d err = estimate.inverse() * gt;
    const Eigen::Matrix3d R_err = err.block<3,3>(0,0);
    const Eigen::Vector3d t_err = err.block<3,1>(0,3);
    return cvo::dist_se3<double>(R_err, t_err);
}

std::vector<double> evaluateErrors(const cvo::pgo::MapOfPoses& poses,
                                   const std::vector<Eigen::Matrix4f>& gt_poses) {
    std::vector<double> errors;
    errors.reserve(gt_poses.size());
    for (int i = 0; i < static_cast<int>(gt_poses.size()); ++i) {
        const Eigen::Matrix4d est =
            cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(poses.at(i));
        errors.push_back(poseDistance(est, gt_poses[i].cast<double>()));
    }
    return errors;
}

double averageNonReferenceError(const std::vector<double>& errors) {
    double sum = 0.0;
    for (size_t i = 1; i < errors.size(); ++i) {
        sum += errors[i];
    }
    return sum / static_cast<double>(errors.size() - 1);
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

void writeStackedCloud(const std::string& filename,
                       const std::vector<CvoCloud>& clouds,
                       const cvo::pgo::MapOfPoses& poses) {
    CvoCloud stacked = stackClouds(clouds, poses);
    const std::filesystem::path output_path = std::filesystem::current_path() / filename;
    const int ret = cvo::point_cloud_io::save_point_cloud(output_path, stacked);
    if (ret != 0) {
        throw std::runtime_error("Failed to write stacked cloud: " + output_path.string());
    }
    std::cout << "Wrote " << output_path << std::endl;
}

cvo::CvoParams makeTestParams(int num_frames,
                              int backend,
                              int linear_backend,
                              int objective_backend,
                              int enable_line_search,
                              int release_binary_state_gpu_each_iter,
                              int stream_frame_clouds_gpu) {
    cvo::CvoParams params;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.multiframe_max_iters = (num_frames == 2) ? 16 : 12;
    params.multiframe_iterations_per_ell = (num_frames == 2) ? 16 : 4;
    params.multiframe_ell_init = 1.0f;
    params.multiframe_ell_min = (num_frames == 2) ? 0.05f : 0.05f;
    params.multiframe_ell_decay_rate = 0.8f;
    params.multiframe_num_neighbors = (num_frames == 2) ? 64 : 48;
    params.multiframe_min_nonzeros = 16;
    params.multiframe_sparse_fill_backend = backend;
    params.multiframe_linear_system_backend = linear_backend;
    params.multiframe_objective_eval_backend = objective_backend;
    params.multiframe_enable_line_search = enable_line_search;
    params.multiframe_release_binary_state_gpu_each_iter = release_binary_state_gpu_each_iter;
    params.multiframe_stream_frame_clouds_gpu = stream_frame_clouds_gpu;
    params.is_using_kdtree = 0;
    params.multiframe_enable_iteration_log = (num_frames == 2) ? 1 : 0;
    params.multiframe_iteration_log_path =
        (std::filesystem::current_path() /
         ("multiframe_bunny_trace_" + std::to_string(num_frames) + ".csv")).string();
    params.multiframe_enable_pose_log = (num_frames == 2) ? 1 : 0;
    params.multiframe_pose_log_path =
        (std::filesystem::current_path() /
         ("multiframe_bunny_pose_trace_" + std::to_string(num_frames) + ".csv")).string();
    return params;
}

}  // namespace

int main(int argc, char** argv) {
    const int num_frames = (argc > 1) ? std::stoi(argv[1]) : 2;
    const int backend = (argc > 2) ? std::stoi(argv[2]) : 1;
    const int linear_backend = (argc > 3) ? std::stoi(argv[3]) : 0;
    const int objective_backend = (argc > 4) ? std::stoi(argv[4]) : 0;
    const int enable_line_search = (argc > 5) ? std::stoi(argv[5]) : 1;
    const int release_binary_state_gpu_each_iter = (argc > 6) ? std::stoi(argv[6]) : 0;
    const int stream_frame_clouds_gpu = (argc > 7) ? std::stoi(argv[7]) : 0;
    if (num_frames < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [num_frames>=2] [backend:0=cpu,1=gpu] [linear_backend:0=cpu,1=gpu]"
                  << " [objective_backend:0=cpu,1=gpu] [enable_line_search:0/1]"
                  << " [release_binary_state_gpu_each_iter:0/1] [stream_frame_clouds_gpu:0/1]" << std::endl;
        return 1;
    }

    const pcl::PointCloud<pcl::PointXYZ> base_cloud = loadBaseCloud();
    std::cout << "Using " << base_cloud.size() << " bunny points per frame across "
              << num_frames << " frames" << std::endl;
    const std::vector<Eigen::Matrix4f> gt_poses = generateGroundTruthPoses(num_frames);
    const std::vector<CvoCloud> clouds = makeObservedClouds(base_cloud, gt_poses);
    const cvo::pgo::MapOfPoses initial_poses = makeIdentityPoses(num_frames);
    const cvo::pgo::VectorOfConstraints constraints = makeAllPairConstraints(gt_poses);

    const std::vector<double> before_errors = evaluateErrors(initial_poses, gt_poses);
    writeStackedCloud("multiframe_bunny_init_stack_" + std::to_string(num_frames) + ".ply",
                      clouds, initial_poses);

    cvo::CvoGPU<PointType> solver(
        makeTestParams(num_frames, backend, linear_backend, objective_backend, enable_line_search,
                       release_binary_state_gpu_each_iter, stream_frame_clouds_gpu));
    cvo::pgo::MapOfPoses optimized;
    double registration_seconds = 0.0;
    const int ret = solver.align_multiframe(
        clouds, initial_poses, constraints, &optimized, &registration_seconds);
    assert(ret == 0);

    const std::vector<double> after_errors = evaluateErrors(optimized, gt_poses);
    writeStackedCloud("multiframe_bunny_final_stack_" + std::to_string(num_frames) + ".ply",
                      clouds, optimized);

    const double avg_before = averageNonReferenceError(before_errors);
    const double avg_after = averageNonReferenceError(after_errors);

    std::cout << "Backend: " << backend
              << " linear_backend=" << linear_backend
              << " objective_backend=" << objective_backend
              << " line_search=" << enable_line_search
              << " release_binary_state_gpu_each_iter=" << release_binary_state_gpu_each_iter
              << " stream_frame_clouds_gpu=" << stream_frame_clouds_gpu << '\n'
              << "Registration seconds: " << registration_seconds << '\n'
              << "Average non-reference pose error before: " << avg_before << '\n'
              << "Average non-reference pose error after: " << avg_after << std::endl;

    for (int i = 1; i < num_frames; ++i) {
        std::cout << "Frame " << i
                  << " before=" << before_errors[i]
                  << " after=" << after_errors[i] << std::endl;
        assert(std::isfinite(after_errors[i]));
        assert(after_errors[i] < before_errors[i]);
    }

    assert(avg_after < avg_before);
    if (num_frames == 2) {
        assert(avg_after < 0.75 * avg_before);
    } else {
        assert(avg_after < 0.2);
    }

    return 0;
}
