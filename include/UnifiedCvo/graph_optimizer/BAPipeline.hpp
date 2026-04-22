#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "cvo/CvoGPU.hpp"
#include "cvo/PoseGraphOptimization.hpp"
#include "dataset_handler/DataHandler.hpp"
#include "utils/DatasetHandlerUtils.hpp"

namespace cvo {

struct BAPipelineOptions {
    std::string dataset_type;
    std::filesystem::path dataset_root;
    std::string params_yaml;
    std::filesystem::path graph_file;
    std::filesystem::path output_prefix;
    std::filesystem::path pose_file;
    std::filesystem::path calibration_file;
};

struct BAGraphSpec {
    std::vector<int> frame_ids;
    std::vector<std::pair<int, int>> edges;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> optional_poses;
};

template <typename PointT>
class BAPipeline {
public:
    using PointType = PointT;
    using CvoCloud = CvoPointCloud<PointType>;

    explicit BAPipeline(BAPipelineOptions options);

    int run();

    static BAGraphSpec read_graph_spec(const std::filesystem::path& path);

private:
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> load_pose_sequence(int max_frame_id) const;
    pgo::MapOfPoses make_initial_poses() const;
    pgo::VectorOfConstraints make_constraints() const;
    void load_graph();
    void load_and_prepare_point_clouds();
    void build_multiframe_problem();
    int run_ba();
    void export_outputs() const;

    static void write_stacked_cloud(const std::filesystem::path& output_path,
                                    const std::vector<CvoCloud>& clouds,
                                    const pgo::MapOfPoses& poses,
                                    const std::vector<int>& frame_ids);

    BAPipelineOptions options_;
    RunnerDatasetType dataset_type_;
    std::unique_ptr<DatasetHandler> dataset_handler_;
    BAGraphSpec graph_;
    std::vector<CvoCloud> clouds_;
    pgo::MapOfPoses initial_poses_;
    pgo::MapOfPoses optimized_poses_;
    pgo::VectorOfConstraints constraints_;
    double registration_seconds_ = 0.0;
};

}  // namespace cvo

#include "graph_optimizer/BAPipeline.tpp"
