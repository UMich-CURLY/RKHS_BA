#include <filesystem>
#include <iostream>
#include <string>

#include "graph_optimizer/BAPipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_type:{pcd|kitti_lidar|tartan_rgbd}>"
                  << " <dataset_root> <params_yaml> <graph_file> <output_prefix>"
                  << " [pose_file] [calibration_file]\n";
        return 1;
    }

    cvo::BAPipelineOptions options;
    options.dataset_type = argv[1];
    options.dataset_root = argv[2];
    options.params_yaml = argv[3];
    options.graph_file = argv[4];
    options.output_prefix = argv[5];
    options.pose_file = (argc > 6) ? std::filesystem::path(argv[6]) : std::filesystem::path();
    options.calibration_file = (argc > 7) ? std::filesystem::path(argv[7]) : std::filesystem::path();

    cvo::BAPipeline<pcl::PointSemantic<3, 19>> pipeline(options);
    return pipeline.run();
}
