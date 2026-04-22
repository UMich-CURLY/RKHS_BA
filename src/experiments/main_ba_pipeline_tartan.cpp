#include <filesystem>
#include <iostream>
#include <string>

#include "graph_optimizer/BAPipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <tartan_traj_root> <params_yaml> <graph_file> <output_prefix>"
                  << " [pose_file] [calibration_file]\n";
        return 1;
    }

    const std::filesystem::path dataset_root = argv[1];

    cvo::BAPipelineOptions options;
    options.dataset_type = "tartan_rgbd";
    options.dataset_root = dataset_root;
    options.params_yaml = argv[2];
    options.graph_file = argv[3];
    options.output_prefix = argv[4];
    options.pose_file = (argc > 5) ? std::filesystem::path(argv[5]) : (dataset_root / "pose_left.txt");
    options.calibration_file = (argc > 6) ? std::filesystem::path(argv[6]) : (dataset_root / "cvo_calib_deep_depth.txt");

    cvo::BAPipeline<pcl::PointSemantic<3, 19>> pipeline(options);
    return pipeline.run();
}
