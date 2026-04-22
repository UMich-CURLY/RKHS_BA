#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace cvo {

  struct CvoParams {
    // ========== Two‑frame parameters ==========
    float ell_init_first_frame = 0.5f;
    float ell_init = 0.5f;
    float ell_min = 0.05f;
    int   min_ell_iter_limit = 1;
    float ell_max = 1.2f;
    double dl = 0.0;
    double dl_step = 0.3;
    float sigma = 0.1f;
    float sp_thres = 0.0006f;
    float c = 7.0f;
    float d = 7.0f;
    float c_ell = 0.15f;
    float c_sigma = 0.6f;
    float s_ell = 0.1f;
    float s_sigma = 0.8f;
    int   MAX_ITER = 10000;
    float eps = 0.00005f;
    float eps_2 = 0.000012f;
    float min_step = 2e-5f;
    float max_step = 0.8f;
    float step = 0.0f;
    float ell_decay_rate = 0.9f;
    float ell_decay_rate_first_frame = 0.99f;
    int   ell_decay_start = 30;
    int   ell_decay_start_first_frame = 300;
    int   indicator_window_size = 15;
    float indicator_stable_threshold = 0.2f;
    int   is_using_geometry = 1;
    int   is_using_intensity = 0;
    int   is_using_semantics = 0;
    int   is_using_range_ell = 0;
    int   is_using_kdtree = 0;
    int   is_using_geometric_type = 0;
    int   is_exporting_association = 0;
    int   is_ell_adaptive = 0;
    int   is_full_ip_matrix = 0;
    int   nearest_neighbors_max = 512;
    int   is_global_angle_registration = 0;

    // ========== Multi‑frame parameters ==========
    int   multiframe_max_iters = 200;
    float multiframe_ell_init = 0.15f;
    float multiframe_ell_min = 0.05f;
    float multiframe_ell_decay_rate = 0.7f;
    int   multiframe_iterations_per_ell = 50;
    float multiframe_downsample_voxel_size = 0.0f;
    int   multiframe_num_neighbors = 128;
    int   multiframe_min_nonzeros = 300;
    int   multiframe_sparse_fill_backend = 1; // 0 = cpu_mutual, 1 = gpu_mutual_knn
    int   multiframe_linear_system_backend = 0; // 0 = cpu_dense, 1 = gpu_assemble_dense, 2 = cpu_sparse_block, 3 = gpu_sparse_block
    int   multiframe_objective_eval_backend = 0; // 0 = cpu_eval, 1 = gpu_eval
    int   multiframe_enable_line_search = 1;
    int   multiframe_release_binary_state_gpu_each_iter = 0;
    int   multiframe_stream_frame_clouds_gpu = 0;
    int   multiframe_enable_iteration_log = 0;
    std::string multiframe_iteration_log_path;
    int   multiframe_enable_pose_log = 0;
    std::string multiframe_pose_log_path;

    CvoParams() = default;
  };

  // YAML reader (kept for convenience)
  inline void read_CvoParams_yaml(const char* filename, CvoParams* params) {
    YAML::Node fs = YAML::LoadFile(filename);
    if (fs["ell_init_first_frame"]) params->ell_init_first_frame = fs["ell_init_first_frame"].as<float>();
    if (fs["ell_init"]) params->ell_init = fs["ell_init"].as<float>();
    if (fs["ell_min"]) params->ell_min = fs["ell_min"].as<float>();
    if (fs["min_ell_iter_limit"]) params->min_ell_iter_limit = fs["min_ell_iter_limit"].as<int>();
    if (fs["ell_max"]) params->ell_max = fs["ell_max"].as<float>();
    if (fs["dl"]) params->dl = fs["dl"].as<double>();
    if (fs["dl_step"]) params->dl_step = fs["dl_step"].as<double>();
    if (fs["sigma"]) params->sigma = fs["sigma"].as<float>();
    if (fs["sp_thres"]) params->sp_thres = fs["sp_thres"].as<float>();
    if (fs["c"]) params->c = fs["c"].as<float>();
    if (fs["d"]) params->d = fs["d"].as<float>();
    if (fs["c_ell"]) params->c_ell = fs["c_ell"].as<float>();
    if (fs["c_sigma"]) params->c_sigma = fs["c_sigma"].as<float>();
    if (fs["s_ell"]) params->s_ell = fs["s_ell"].as<float>();
    if (fs["s_sigma"]) params->s_sigma = fs["s_sigma"].as<float>();
    if (fs["MAX_ITER"]) params->MAX_ITER = fs["MAX_ITER"].as<int>();
    if (fs["eps"]) params->eps = fs["eps"].as<float>();
    if (fs["eps_2"]) params->eps_2 = fs["eps_2"].as<float>();
    if (fs["min_step"]) params->min_step = fs["min_step"].as<float>();
    if (fs["max_step"]) params->max_step = fs["max_step"].as<float>();
    if (fs["ell_decay_rate"]) params->ell_decay_rate = fs["ell_decay_rate"].as<float>();
    if (fs["ell_decay_rate_first_frame"]) params->ell_decay_rate_first_frame = fs["ell_decay_rate_first_frame"].as<float>();
    if (fs["ell_decay_start"]) params->ell_decay_start = fs["ell_decay_start"].as<int>();
    if (fs["ell_decay_start_first_frame"]) params->ell_decay_start_first_frame = fs["ell_decay_start_first_frame"].as<int>();
    if (fs["indicator_window_size"]) params->indicator_window_size = fs["indicator_window_size"].as<int>();
    if (fs["indicator_stable_threshold"]) params->indicator_stable_threshold = fs["indicator_stable_threshold"].as<float>();
    if (fs["is_using_geometry"]) params->is_using_geometry = fs["is_using_geometry"].as<int>();
    if (fs["is_using_intensity"]) params->is_using_intensity = fs["is_using_intensity"].as<int>();
    if (fs["is_using_semantics"]) params->is_using_semantics = fs["is_using_semantics"].as<int>();
    if (fs["is_using_range_ell"]) params->is_using_range_ell = fs["is_using_range_ell"].as<int>();
    if (fs["is_using_kdtree"]) params->is_using_kdtree = fs["is_using_kdtree"].as<int>();
    if (fs["is_using_geometric_type"]) params->is_using_geometric_type = fs["is_using_geometric_type"].as<int>();
    if (fs["is_global_angle_registration"]) params->is_global_angle_registration = fs["is_global_angle_registration"].as<int>();
    if (fs["is_exporting_association"]) params->is_exporting_association = fs["is_exporting_association"].as<int>();
    if (fs["nearest_neighbors_max"]) params->nearest_neighbors_max = fs["nearest_neighbors_max"].as<int>();
    if (fs["multiframe_max_iters"]) params->multiframe_max_iters = fs["multiframe_max_iters"].as<int>();
    if (fs["multiframe_ell_init"]) params->multiframe_ell_init = fs["multiframe_ell_init"].as<float>();
    if (fs["multiframe_ell_min"]) params->multiframe_ell_min = fs["multiframe_ell_min"].as<float>();
    if (fs["multiframe_ell_decay_rate"]) params->multiframe_ell_decay_rate = fs["multiframe_ell_decay_rate"].as<float>();
    if (fs["multiframe_iterations_per_ell"]) params->multiframe_iterations_per_ell = fs["multiframe_iterations_per_ell"].as<int>();
    if (fs["multiframe_downsample_voxel_size"]) params->multiframe_downsample_voxel_size = fs["multiframe_downsample_voxel_size"].as<float>();
    if (fs["multiframe_num_neighbors"]) params->multiframe_num_neighbors = fs["multiframe_num_neighbors"].as<int>();
    if (fs["multiframe_min_nonzeros"]) params->multiframe_min_nonzeros = fs["multiframe_min_nonzeros"].as<int>();
    if (fs["multiframe_sparse_fill_backend"]) params->multiframe_sparse_fill_backend = fs["multiframe_sparse_fill_backend"].as<int>();
    if (fs["multiframe_linear_system_backend"]) params->multiframe_linear_system_backend = fs["multiframe_linear_system_backend"].as<int>();
    if (fs["multiframe_objective_eval_backend"]) params->multiframe_objective_eval_backend = fs["multiframe_objective_eval_backend"].as<int>();
    if (fs["multiframe_enable_line_search"]) params->multiframe_enable_line_search = fs["multiframe_enable_line_search"].as<int>();
    if (fs["multiframe_release_binary_state_gpu_each_iter"]) params->multiframe_release_binary_state_gpu_each_iter = fs["multiframe_release_binary_state_gpu_each_iter"].as<int>();
    if (fs["multiframe_stream_frame_clouds_gpu"]) params->multiframe_stream_frame_clouds_gpu = fs["multiframe_stream_frame_clouds_gpu"].as<int>();
    if (fs["multiframe_enable_iteration_log"]) params->multiframe_enable_iteration_log = fs["multiframe_enable_iteration_log"].as<int>();
    if (fs["multiframe_iteration_log_path"]) params->multiframe_iteration_log_path = fs["multiframe_iteration_log_path"].as<std::string>();
    if (fs["multiframe_enable_pose_log"]) params->multiframe_enable_pose_log = fs["multiframe_enable_pose_log"].as<int>();
    if (fs["multiframe_pose_log_path"]) params->multiframe_pose_log_path = fs["multiframe_pose_log_path"].as<std::string>();
  }

} // namespace cvo
