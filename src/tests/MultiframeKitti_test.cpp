#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "cvo/CvoGPU.cuh"
#include "cvo/PoseGraphOptimization.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "utils/CvoPointCloud.hpp"

using PointType = pcl::PointSemantic<1, 19>;
using CvoCloud = cvo::CvoPointCloud<PointType>;

namespace {

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
load_kitti_poses(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("Failed to open pose file: " + path.string());
    }

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        double pose_vec[12];
        for (double& value : pose_vec) {
            ss >> value;
        }
        if (!ss.good() && !ss.eof()) {
            throw std::runtime_error("Failed to parse pose line from: " + path.string());
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 4>(0, 0) =
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(pose_vec);
        poses.push_back(T);
    }
    return poses;
}

double pose_distance(const Eigen::Matrix4d& estimate, const Eigen::Matrix4d& gt) {
    const Eigen::Matrix4d err = estimate.inverse() * gt;
    const Eigen::Matrix3d R_err = err.block<3, 3>(0, 0);
    const Eigen::Vector3d t_err = err.block<3, 1>(0, 3);
    return cvo::dist_se3<double>(R_err, t_err);
}

std::vector<double> evaluate_errors(
    const cvo::pgo::MapOfPoses& poses,
    const std::vector<int>& frame_ids,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& gt_poses) {
    std::vector<double> errors;
    errors.reserve(frame_ids.size());
    for (const int id : frame_ids) {
        const Eigen::Matrix4d est = cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(poses.at(id));
        errors.push_back(pose_distance(est, gt_poses.at(id)));
    }
    return errors;
}

double average_non_reference_error(const std::vector<double>& errors) {
    double sum = 0.0;
    for (size_t i = 1; i < errors.size(); ++i) {
        sum += errors[i];
    }
    return sum / static_cast<double>(errors.size() - 1);
}

std::vector<CvoCloud> load_kitti_clouds(const std::filesystem::path& seq_root,
                                        const std::vector<int>& frame_ids,
                                        float voxel_size) {
    cvo::KittiHandler handler(seq_root.string(),
                              cvo::KittiHandler::DataType::LIDAR,
                              cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME);
    std::vector<CvoCloud> clouds;
    clouds.reserve(frame_ids.size());

    for (const int frame_id : frame_ids) {
        handler.set_start_index(frame_id);
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw(new pcl::PointCloud<pcl::PointXYZI>);
        if (handler.read_next_lidar(raw) != 0) {
            throw std::runtime_error("Failed to read KITTI lidar frame " + std::to_string(frame_id));
        }

        pcl::PointCloud<pcl::PointXYZI> downsampled;
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setInputCloud(raw);
        voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel.filter(downsampled);

        std::cout << "Frame " << frame_id
                  << " raw_points=" << raw->size()
                  << " downsampled_points=" << downsampled.size()
                  << std::endl;
        clouds.emplace_back(downsampled);
    }
    return clouds;
}

cvo::pgo::MapOfPoses make_initial_poses(
    const std::vector<int>& frame_ids,
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& init_poses) {
    cvo::pgo::MapOfPoses poses;
    for (const int id : frame_ids) {
        poses[id] = cvo::pgo::pose3d_from_eigen(init_poses.at(id));
    }
    return poses;
}

cvo::pgo::VectorOfConstraints make_all_pair_constraints(const std::vector<int>& frame_ids) {
    cvo::pgo::VectorOfConstraints constraints;
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        for (size_t j = i + 1; j < frame_ids.size(); ++j) {
            cvo::pgo::Constraint3d con;
            con.id_begin = frame_ids[i];
            con.id_end = frame_ids[j];
            con.t_be = cvo::pgo::pose3d_from_eigen(Eigen::Matrix4d::Identity());
            con.information = Eigen::Matrix<double, 6, 6>::Identity();
            constraints.push_back(con);
        }
    }
    return constraints;
}

cvo::CvoParams make_test_params(const std::filesystem::path& trace_path,
                                const std::filesystem::path& pose_trace_path) {
    cvo::CvoParams params;
    params.sigma = 0.1f;
    params.sp_thres = 0.006f;
    params.c_ell = 0.1f;
    params.c_sigma = 1.0f;
    params.is_using_geometry = 1;
    params.is_using_intensity = 1;
    params.is_using_semantics = 0;
    params.is_using_geometric_type = 0;
    params.is_using_kdtree = 0;
    params.multiframe_sparse_fill_backend = 1;
    params.multiframe_max_iters = 5;
    params.multiframe_ell_init = 2.0f;
    params.multiframe_ell_min = 0.5f;
    params.multiframe_ell_decay_rate = 0.9f;
    params.multiframe_iterations_per_ell = 4;
    params.multiframe_num_neighbors = 12;
    params.multiframe_min_nonzeros = 20;
    params.multiframe_enable_iteration_log = 1;
    params.multiframe_iteration_log_path = trace_path.string();
    params.multiframe_enable_pose_log = 1;
    params.multiframe_pose_log_path = pose_trace_path.string();
    params.multiframe_linear_system_backend = 1;
    return params;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path seq_root =
        (argc > 1) ? std::filesystem::path(argv[1])
                   : std::filesystem::path("/run/media/rayzhang/Samsung_T5/kitti_lidar/dataset/sequences/05");
    //const std::filesystem::path init_pose_path =
    //    (argc > 2) ? std::filesystem::path(argv[2]) : (seq_root / "poses.txt");
    const std::filesystem::path gt_pose_path =
        (argc > 2) ? std::filesystem::path(argv[2])
                   : std::filesystem::path("../RKHS_BA/ground_truth/kitti/lidar/05.txt");
    const float voxel_size = (argc > 3) ? std::stof(argv[3]) : 1.0f;

    const std::vector<int> frame_ids = {0, 1, 2 , 3, 4, 5};
    const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> init_poses(frame_ids.size(), Eigen::Matrix4d::Identity());
    //load_kitti_poses(init_pose_path);
    const auto gt_poses = load_kitti_poses(gt_pose_path);
    const auto clouds = load_kitti_clouds(seq_root, frame_ids, voxel_size);
    const auto initial_poses = make_initial_poses(frame_ids, init_poses);
    const auto constraints = make_all_pair_constraints(frame_ids);

    const auto before_errors = evaluate_errors(initial_poses, frame_ids, gt_poses);

    const std::filesystem::path trace_path =
        std::filesystem::current_path() / "multiframe_kitti_seq05_trace.csv";
    const std::filesystem::path pose_trace_path =
        std::filesystem::current_path() / "multiframe_kitti_seq05_pose_trace.csv";

    cvo::CvoGPU<PointType> solver(make_test_params(trace_path, pose_trace_path));
    cvo::pgo::MapOfPoses optimized;
    double registration_seconds = 0.0;
    const int ret = solver.align_multiframe(
        clouds, initial_poses, constraints, &optimized, &registration_seconds);
    assert(ret == 0);

    const auto after_errors = evaluate_errors(optimized, frame_ids, gt_poses);
    const double avg_before = average_non_reference_error(before_errors);
    const double avg_after = average_non_reference_error(after_errors);

    std::cout << "Registration seconds: " << registration_seconds << '\n'
              << "Iteration trace: " << trace_path << '\n'
              << "Pose trace: " << pose_trace_path << '\n'
              << "Average non-reference pose error before: " << avg_before << '\n'
              << "Average non-reference pose error after: " << avg_after << std::endl;
    for (size_t i = 0; i < frame_ids.size(); ++i) {
        std::cout << "Frame " << frame_ids[i]
                  << " before=" << before_errors[i]
                  << " after=" << after_errors[i] << std::endl;
    }

    assert(std::isfinite(avg_after));
    return 0;
}
