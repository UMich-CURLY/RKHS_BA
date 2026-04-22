#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "utils/DatasetHandlerUtils.hpp"
#include "utils/PointCloudLoader.hpp"

namespace {

using SemanticPoint = pcl::PointSemantic<3, 19>;
using Cloud = cvo::CvoPointCloud<SemanticPoint>;

void test_pcd_sequence(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);

    pcl::PointCloud<pcl::PointXYZ> cloud0;
    cloud0.push_back(pcl::PointXYZ{1.0f, 2.0f, 3.0f});
    cloud0.width = 1;
    cloud0.height = 1;
    pcl::PointCloud<pcl::PointXYZ> cloud1;
    cloud1.push_back(pcl::PointXYZ{4.0f, 5.0f, 6.0f});
    cloud1.width = 1;
    cloud1.height = 1;
    cvo::point_cloud_io::save_point_cloud(root / "000000.pcd", cloud0);
    cvo::point_cloud_io::save_point_cloud(root / "000001.pcd", cloud1);

    auto handler = cvo::create_dataset_handler("pcd", root);
    Cloud loaded;
    cvo::load_cloud_from_dataset_handler(*handler, "pcd", 1, loaded);
    assert(loaded.size() == 1);
    assert(std::abs(loaded[0].x - 4.0f) < 1e-6f);
}

void test_kitti_lidar(const std::filesystem::path& root) {
    const std::filesystem::path velodyne = root / "velodyne";
    std::filesystem::create_directories(velodyne);

    const std::vector<float> raw = {
        1.0f, 2.0f, 3.0f, 0.5f,
        4.0f, 5.0f, 6.0f, 0.75f,
    };
    std::ofstream out(velodyne / "000000.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    out.close();

    auto handler = cvo::create_dataset_handler("kitti_lidar", root);
    Cloud loaded;
    cvo::load_cloud_from_dataset_handler(*handler, "kitti_lidar", 0, loaded);
    assert(loaded.size() == 2);
    assert(std::isfinite(loaded[0].x));
    assert(std::isfinite(loaded[0].y));
    assert(std::isfinite(loaded[0].z));
    assert(std::isfinite(loaded[1].x));
    assert(std::isfinite(loaded[1].y));
    assert(std::isfinite(loaded[1].z));
    assert(loaded[1].x > loaded[0].x);
    assert(loaded[1].y > loaded[0].y);
    assert(loaded[1].z > loaded[0].z);
}

void test_tartan_rgbd(const std::filesystem::path& root) {
    const std::filesystem::path image_dir = root / "image_left";
    const std::filesystem::path depth_dir = root / "deep_depth";
    std::filesystem::create_directories(image_dir);
    std::filesystem::create_directories(depth_dir);

    std::ofstream calib(root / "cvo_calib_deep_depth.txt");
    calib << "100 100 0 0 100\n";
    calib.close();

    cv::Mat rgb(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
    rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(70, 80, 90);
    rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(100, 110, 120);
    cv::imwrite((image_dir / "000000_left.png").string(), rgb);

    cv::Mat depth(2, 2, CV_16UC1, cv::Scalar(0));
    depth.at<std::uint16_t>(0, 0) = 100;
    depth.at<std::uint16_t>(0, 1) = 200;
    depth.at<std::uint16_t>(1, 0) = 0;
    depth.at<std::uint16_t>(1, 1) = 300;
    cv::imwrite((depth_dir / "000000_left_depth.png").string(), depth);

    Cloud loaded;
    auto handler = cvo::create_dataset_handler("tartan_rgbd", root);
    cvo::load_cloud_from_dataset_handler(*handler, "tartan_rgbd", 0, loaded, root / "cvo_calib_deep_depth.txt", 10.0f);
    assert(loaded.size() == 3);
    assert(std::abs(loaded[0].z - 1.0f) < 1e-6f);
    assert(loaded[0].r == 30);
    assert(loaded[0].g == 20);
    assert(loaded[0].b == 10);
}

}  // namespace

int main() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "cvo_dataset_loader_test";
    std::filesystem::remove_all(root);

    test_pcd_sequence(root / "pcd");
    test_kitti_lidar(root / "kitti");
    test_tartan_rgbd(root / "tartan");

    std::cout << "Dataset handler tests passed." << std::endl;
    return 0;
}
