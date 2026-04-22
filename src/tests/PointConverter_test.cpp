#include <cassert>
#include <iostream>

#include <pcl/point_types.h>

#include "utils/CvoPointCloud.hpp"
#include "utils/PointConverter.hpp"

using SemanticPoint = pcl::PointSemantic<3, 19>;

int main() {
    pcl::PointCloud<pcl::PointXYZ> xyz_cloud;
    pcl::PointXYZ xyz;
    xyz.x = 1.0f;
    xyz.y = -2.0f;
    xyz.z = 3.5f;
    xyz_cloud.push_back(xyz);

    cvo::CvoPointCloud<SemanticPoint> semantic_cloud(xyz_cloud);
    assert(semantic_cloud.size() == 1);
    assert(semantic_cloud[0].x == 1.0f);
    assert(semantic_cloud[0].y == -2.0f);
    assert(semantic_cloud[0].z == 3.5f);

    pcl::PointCloud<SemanticPoint> semantic_pcl;
    semantic_pcl.push_back(semantic_cloud[0]);
    pcl::PointCloud<pcl::PointXYZRGB> xyzrgb_cloud;
    cvo::point_converter::convert_point_cloud(semantic_pcl, xyzrgb_cloud);
    assert(xyzrgb_cloud.size() == 1);
    assert(xyzrgb_cloud[0].x == 1.0f);
    assert(xyzrgb_cloud[0].y == -2.0f);
    assert(xyzrgb_cloud[0].z == 3.5f);

    pcl::PointXYZRGB rgb;
    rgb.x = 0.5f;
    rgb.y = 0.25f;
    rgb.z = -1.0f;
    rgb.r = 10;
    rgb.g = 20;
    rgb.b = 30;
    SemanticPoint semantic_from_rgb;
    cvo::point_converter::convert_point(rgb, semantic_from_rgb);
    assert(semantic_from_rgb.x == rgb.x);
    assert(semantic_from_rgb.y == rgb.y);
    assert(semantic_from_rgb.z == rgb.z);
    assert(semantic_from_rgb.r == rgb.r);
    assert(semantic_from_rgb.g == rgb.g);
    assert(semantic_from_rgb.b == rgb.b);

    std::cout << "PointConverter tests passed." << std::endl;
    return 0;
}
