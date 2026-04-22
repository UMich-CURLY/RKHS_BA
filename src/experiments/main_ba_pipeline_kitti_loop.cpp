#include <filesystem>
#include <iostream>
#include <string>

#include "graph_optimizer/BAPipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <kitti_sequence_root> <params_yaml> <graph_file> <output_prefix> <tracking_pose_file>"
                  << " [calibration_file]\n";
        return 1;
    }

    cvo::BAPipelineOptions options;
    options.dataset_type = "kitti_lidar";
    options.dataset_root = argv[1];
    options.params_yaml = argv[2];
    options.graph_file = argv[3];
    options.output_prefix = argv[4];
    options.pose_file = argv[5];
    options.calibration_file = (argc > 6) ? std::filesystem::path(argv[6]) : std::filesystem::path();

    cvo::BAPipeline<pcl::PointSemantic<1, 19>> pipeline(options);
    return pipeline.run();
}
