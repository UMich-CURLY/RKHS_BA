#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "utils/PointCloudLoader.hpp"
#include "utils/PointSemantic.hpp"

namespace {

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_xyz_round_trip() {
    pcl::PointCloud<pcl::PointXYZ> input;
    input.push_back(pcl::PointXYZ{1.0f, 2.0f, 3.0f});
    input.push_back(pcl::PointXYZ{-0.5f, 4.0f, 1.25f});
    input.width = static_cast<std::uint32_t>(input.size());
    input.height = 1;

    const std::filesystem::path path = "point_loader_xyz_test.ply";
    assert(cvo::point_cloud_io::save_point_cloud(path, input) == 0);

    cvo::CvoPointCloud<pcl::PointSemantic<3, 19>> loaded;
    assert(cvo::point_cloud_io::load_point_cloud(path, loaded) == 0);
    assert(loaded.size() == input.size());
    assert(nearly_equal(loaded[0].x, input[0].x));
    assert(nearly_equal(loaded[0].y, input[0].y));
    assert(nearly_equal(loaded[0].z, input[0].z));

    std::filesystem::remove(path);
}

void test_rgb_round_trip() {
    pcl::PointCloud<pcl::PointXYZRGB> input;
    pcl::PointXYZRGB p0;
    p0.x = 0.1f; p0.y = -0.2f; p0.z = 0.3f; p0.r = 7; p0.g = 21; p0.b = 99;
    pcl::PointXYZRGB p1;
    p1.x = -1.0f; p1.y = 2.5f; p1.z = 4.0f; p1.r = 200; p1.g = 111; p1.b = 3;
    input.push_back(p0);
    input.push_back(p1);
    input.width = static_cast<std::uint32_t>(input.size());
    input.height = 1;

    const std::filesystem::path path = "point_loader_rgb_test.ply";
    assert(cvo::point_cloud_io::save_point_cloud(path, input) == 0);

    pcl::PointCloud<pcl::PointXYZRGB> loaded;
    assert(cvo::point_cloud_io::load_point_cloud(path, loaded) == 0);
    assert(loaded.size() == input.size());
    assert(loaded[1].r == input[1].r);
    assert(loaded[1].g == input[1].g);
    assert(loaded[1].b == input[1].b);

    std::filesystem::remove(path);
}

void test_intensity_round_trip() {
    pcl::PointCloud<pcl::PointXYZI> input;
    pcl::PointXYZI p0;
    p0.x = 0.0f; p0.y = 1.0f; p0.z = 2.0f; p0.intensity = 0.75f;
    pcl::PointXYZI p1;
    p1.x = -1.0f; p1.y = -2.0f; p1.z = 3.5f; p1.intensity = 5.0f;
    input.push_back(p0);
    input.push_back(p1);
    input.width = static_cast<std::uint32_t>(input.size());
    input.height = 1;

    const std::filesystem::path path = "point_loader_intensity_test.ply";
    assert(cvo::point_cloud_io::save_point_cloud(path, input) == 0);

    pcl::PointCloud<pcl::PointXYZI> loaded;
    assert(cvo::point_cloud_io::load_point_cloud(path, loaded) == 0);
    assert(loaded.size() == input.size());
    assert(nearly_equal(loaded[0].intensity, input[0].intensity));
    assert(nearly_equal(loaded[1].intensity, input[1].intensity));

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_xyz_round_trip();
    test_rgb_round_trip();
    test_intensity_round_trip();
    std::cout << "PointCloudLoader tests passed." << std::endl;
    return 0;
}
