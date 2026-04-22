#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "cvo/CvoParams.hpp"
#include "utils/PointCloudLoader.hpp"

namespace cvo {

template <typename PointT>
BAPipeline<PointT>::BAPipeline(BAPipelineOptions options)
    : options_(std::move(options)),
      dataset_type_(parse_runner_dataset_type(options_.dataset_type)) {}

template <typename PointT>
BAGraphSpec BAPipeline<PointT>::read_graph_spec(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("Failed to open graph file: " + path.string());
    }

    BAGraphSpec graph;
    int num_frames = 0;
    int num_edges = 0;
    in >> num_frames >> num_edges;
    graph.frame_ids.resize(num_frames);
    for (int i = 0; i < num_frames; ++i) {
        in >> graph.frame_ids[i];
    }
    graph.edges.resize(num_edges);
    for (int i = 0; i < num_edges; ++i) {
        in >> graph.edges[i].first >> graph.edges[i].second;
    }

    while (in.good()) {
        double pose_vec[12];
        for (double& value : pose_vec) {
            in >> value;
        }
        if (!in.good()) {
            break;
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 4>(0, 0) =
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(pose_vec);
        graph.optional_poses.push_back(T);
    }
    return graph;
}

template <typename PointT>
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
BAPipeline<PointT>::load_pose_sequence(int max_frame_id) const {
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
    if (options_.pose_file.empty()) {
        return poses;
    }

    switch (dataset_type_) {
        case RunnerDatasetType::TartanAirRgbd: {
            std::ifstream in(options_.pose_file);
            std::string line;
            int line_index = 0;
            while (std::getline(in, line)) {
                if (line_index > max_frame_id) {
                    break;
                }
                std::stringstream line_stream(line);
                double xyz[3];
                double q[4];
                line_stream >> xyz[0] >> xyz[1] >> xyz[2] >> q[0] >> q[1] >> q[2] >> q[3];
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3, 3>(0, 0) =
                    Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
                T.block<3, 1>(0, 3) = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
                poses.push_back(T);
                ++line_index;
            }
            break;
        }
        case RunnerDatasetType::KittiLidar: {
            std::ifstream in(options_.pose_file);
            std::string line;
            int line_index = 0;
            while (std::getline(in, line)) {
                if (line_index > max_frame_id) {
                    break;
                }
                std::stringstream line_stream(line);
                double pose_vec[12];
                for (double& value : pose_vec) {
                    line_stream >> value;
                }
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3, 4>(0, 0) =
                    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(pose_vec);
                poses.push_back(T);
                ++line_index;
            }
            break;
        }
        case RunnerDatasetType::PcdSequence:
            break;
    }
    return poses;
}

template <typename PointT>
pgo::MapOfPoses BAPipeline<PointT>::make_initial_poses() const {
    pgo::MapOfPoses poses;
    if (!graph_.optional_poses.empty()) {
        for (size_t i = 0; i < graph_.frame_ids.size(); ++i) {
            poses.emplace(graph_.frame_ids[i], pgo::pose3d_from_eigen(graph_.optional_poses.at(i)));
        }
        return poses;
    }

    const int max_frame_id = *std::max_element(graph_.frame_ids.begin(), graph_.frame_ids.end());
    const auto loaded_poses = load_pose_sequence(max_frame_id);
    for (const int frame_id : graph_.frame_ids) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        if (!loaded_poses.empty() && frame_id >= 0 && frame_id < static_cast<int>(loaded_poses.size())) {
            T = loaded_poses[frame_id];
        }
        poses.emplace(frame_id, pgo::pose3d_from_eigen(T));
    }
    return poses;
}

template <typename PointT>
pgo::VectorOfConstraints BAPipeline<PointT>::make_constraints() const {
    pgo::VectorOfConstraints constraints;
    constraints.reserve(graph_.edges.size());
    for (const auto& edge : graph_.edges) {
        pgo::Constraint3d constraint;
        constraint.id_begin = edge.first;
        constraint.id_end = edge.second;
        constraint.t_be = pgo::pose3d_from_eigen(Eigen::Matrix4d::Identity());
        constraint.information = Eigen::Matrix<double, 6, 6>::Identity();
        constraints.push_back(constraint);
    }
    return constraints;
}

template <typename PointT>
void BAPipeline<PointT>::load_graph() {
    std::cout << "[BAPipeline] Loading graph from " << options_.graph_file << std::endl;
    graph_ = read_graph_spec(options_.graph_file);
    std::cout << "[BAPipeline] Graph loaded: frames=" << graph_.frame_ids.size()
              << " edges=" << graph_.edges.size()
              << " embedded_poses=" << graph_.optional_poses.size() << std::endl;
}

template <typename PointT>
void BAPipeline<PointT>::load_and_prepare_point_clouds() {
    CvoParams params;
    read_CvoParams_yaml(options_.params_yaml.c_str(), &params);

    std::cout << "[BAPipeline] Creating dataset handler for " << options_.dataset_type
              << " at " << options_.dataset_root << std::endl;
    dataset_handler_ = create_dataset_handler(options_.dataset_type, options_.dataset_root);
    clouds_.resize(graph_.frame_ids.size());
    std::cout << "[BAPipeline] Loading " << clouds_.size()
              << " point clouds with voxel="
              << params.multiframe_downsample_voxel_size << std::endl;
    for (size_t i = 0; i < graph_.frame_ids.size(); ++i) {
        std::filesystem::path calibration_file = options_.calibration_file;
        if (dataset_type_ == RunnerDatasetType::TartanAirRgbd && calibration_file.empty()) {
            calibration_file = options_.dataset_root / "cvo_calib_deep_depth.txt";
        }
        load_cloud_from_dataset_handler(*dataset_handler_,
                                        options_.dataset_type,
                                        graph_.frame_ids[i],
                                        clouds_[i],
                                        calibration_file,
                                        std::numeric_limits<float>::max(),
                                        params.multiframe_downsample_voxel_size);
        const size_t loaded = i + 1;
        if (loaded <= 5 || loaded == clouds_.size() || loaded % 50 == 0) {
            std::cout << "[BAPipeline] Loaded frame " << graph_.frame_ids[i]
                      << " (" << loaded << "/" << clouds_.size() << ")"
                      << " points=" << clouds_[i].size() << std::endl;
        }
    }
    std::cout << "[BAPipeline] Point cloud loading complete." << std::endl;
}

template <typename PointT>
void BAPipeline<PointT>::build_multiframe_problem() {
    std::cout << "[BAPipeline] Building multiframe problem." << std::endl;
    initial_poses_ = make_initial_poses();
    constraints_ = make_constraints();
    std::cout << "[BAPipeline] Initial poses=" << initial_poses_.size()
              << " constraints=" << constraints_.size() << std::endl;
}

template <typename PointT>
int BAPipeline<PointT>::run_ba() {
    CvoParams params;
    read_CvoParams_yaml(options_.params_yaml.c_str(), &params);
    std::cout << "[BAPipeline] Starting BA:"
              << " linear_backend=" << params.multiframe_linear_system_backend
              << " sparse_fill_backend=" << params.multiframe_sparse_fill_backend
              << " objective_backend=" << params.multiframe_objective_eval_backend
              << " line_search=" << params.multiframe_enable_line_search
              << " ell_init=" << params.multiframe_ell_init
              << " ell_min=" << params.multiframe_ell_min
              << " iterations_per_ell=" << params.multiframe_iterations_per_ell
              << " max_ell_levels=" << params.multiframe_max_iters
              << std::endl;
    CvoGPU<PointType> solver(params);
    const int ret = solver.align_multiframe(clouds_, initial_poses_, constraints_,
                                            &optimized_poses_, &registration_seconds_);
    std::cout << "[BAPipeline] BA finished: ret=" << ret
              << " registration_seconds=" << registration_seconds_ << std::endl;
    return ret;
}

template <typename PointT>
void BAPipeline<PointT>::write_stacked_cloud(const std::filesystem::path& output_path,
                                             const std::vector<CvoCloud>& clouds,
                                             const pgo::MapOfPoses& poses,
                                             const std::vector<int>& frame_ids) {
    pcl::PointCloud<PointType> stacked;
    for (size_t i = 0; i < clouds.size(); ++i) {
        const auto pose_it = poses.find(frame_ids[i]);
        if (pose_it == poses.end()) {
            continue;
        }
        const Eigen::Matrix4f T = pgo::pose3d_to_eigen<float>(pose_it->second);
        CvoCloud transformed;
        CvoCloud::transform(T, clouds[i], transformed);
        for (const auto& point : transformed.points()) {
            stacked.push_back(point);
        }
    }
    stacked.width = static_cast<std::uint32_t>(stacked.size());
    stacked.height = 1;
    stacked.is_dense = false;
    point_cloud_io::save_point_cloud(output_path, stacked);
}

template <typename PointT>
void BAPipeline<PointT>::export_outputs() const {
    std::cout << "[BAPipeline] Exporting trajectory and stacked clouds with prefix "
              << options_.output_prefix << std::endl;
    std::ofstream traj_out(options_.output_prefix.string() + "_trajectory.txt");
    for (const auto& [id, pose] : optimized_poses_) {
        const Eigen::Matrix4d T = pgo::pose3d_to_eigen<double>(pose);
        traj_out << id;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                traj_out << ' ' << T(r, c);
            }
        }
        traj_out << '\n';
    }
    traj_out.close();

    write_stacked_cloud(options_.output_prefix.string() + "_stacked_init.ply", clouds_, initial_poses_, graph_.frame_ids);
    write_stacked_cloud(options_.output_prefix.string() + "_stacked_final.ply", clouds_, optimized_poses_, graph_.frame_ids);
    std::cout << "[BAPipeline] Export complete." << std::endl;
}

template <typename PointT>
int BAPipeline<PointT>::run() {
    load_graph();
    load_and_prepare_point_clouds();
    build_multiframe_problem();
    const int ret = run_ba();
    if (ret != 0) {
        return ret;
    }
    export_outputs();
    return 0;
}

}  // namespace cvo
