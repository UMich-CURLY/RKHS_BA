#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "utils/CvoPointCloud.hpp"
#include "utils/PointConverter.hpp"

namespace cvo {

namespace point_cloud_io {

template <typename PointT>
struct proxy_point_type {
    using type = std::conditional_t<
        point_converter::has_r<PointT>::value &&
            point_converter::has_g<PointT>::value &&
            point_converter::has_b<PointT>::value,
        pcl::PointXYZRGB,
        std::conditional_t<
            point_converter::has_intensity<PointT>::value,
            pcl::PointXYZI,
            pcl::PointXYZ>>;
};

template <typename PointT>
using proxy_point_type_t = typename proxy_point_type<PointT>::type;

inline std::string lowercase_extension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

template <typename PointT>
int load_point_cloud(const std::filesystem::path& path, pcl::PointCloud<PointT>& cloud) {
    const std::string ext = lowercase_extension(path);
    using ProxyPoint = proxy_point_type_t<PointT>;
    pcl::PointCloud<ProxyPoint> proxy_cloud;

    int ret = -1;
    if (ext == ".ply") {
        ret = pcl::io::loadPLYFile(path.string(), proxy_cloud);
    } else if (ext == ".pcd") {
        ret = pcl::io::loadPCDFile(path.string(), proxy_cloud);
    } else {
        throw std::runtime_error("Unsupported point cloud extension: " + path.string());
    }

    if (ret != 0) {
        return ret;
    }

    point_converter::convert_point_cloud(proxy_cloud, cloud);
    return 0;
}

template <typename PointT>
int load_point_cloud(const std::filesystem::path& path, CvoPointCloud<PointT>& cloud) {
    pcl::PointCloud<PointT> pcl_cloud;
    const int ret = load_point_cloud(path, pcl_cloud);
    if (ret != 0) {
        return ret;
    }
    cloud = CvoPointCloud<PointT>(pcl_cloud);
    return 0;
}

template <typename PointT>
int save_point_cloud(const std::filesystem::path& path, const pcl::PointCloud<PointT>& cloud) {
    const std::string ext = lowercase_extension(path);
    using ProxyPoint = proxy_point_type_t<PointT>;
    pcl::PointCloud<ProxyPoint> proxy_cloud;
    point_converter::convert_point_cloud(cloud, proxy_cloud);

    if (ext == ".ply") {
        return pcl::io::savePLYFileBinary(path.string(), proxy_cloud);
    }
    if (ext == ".pcd") {
        return pcl::io::savePCDFileBinary(path.string(), proxy_cloud);
    }
    throw std::runtime_error("Unsupported point cloud extension: " + path.string());
}

template <typename PointT>
int save_point_cloud(const std::filesystem::path& path, const CvoPointCloud<PointT>& cloud) {
    pcl::PointCloud<PointT> pcl_cloud;
    pcl_cloud.reserve(cloud.size());
    for (const auto& point : cloud.points()) {
        pcl_cloud.push_back(point);
    }
    pcl_cloud.width = static_cast<std::uint32_t>(pcl_cloud.size());
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = false;
    return save_point_cloud(path, pcl_cloud);
}

}  // namespace point_cloud_io

}  // namespace cvo
