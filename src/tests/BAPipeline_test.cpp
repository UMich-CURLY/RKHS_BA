#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "graph_optimizer/BAPipeline.hpp"

int main() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "cvo_ba_pipeline_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const std::filesystem::path graph_file = root / "graph.txt";
    {
        std::ofstream out(graph_file);
        out << "3 2\n";
        out << "0 2 5\n";
        out << "0 2\n";
        out << "2 5\n";
        out << "1 0 0 0 0 1 0 0 0 0 1 0\n";
        out << "1 0 0 1 0 1 0 2 0 0 1 3\n";
        out << "1 0 0 4 0 1 0 5 0 0 1 6\n";
    }

    const auto graph = cvo::BAPipeline<pcl::PointSemantic<3, 19>>::read_graph_spec(graph_file);
    assert(graph.frame_ids.size() == 3);
    assert(graph.edges.size() == 2);
    assert(graph.optional_poses.size() == 3);
    assert(graph.frame_ids[1] == 2);
    assert(graph.edges[1].first == 2);
    assert(graph.optional_poses[2](0, 3) == 4.0);

    std::cout << "BAPipeline graph parsing test passed." << std::endl;
    return 0;
}
