#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "dataset_handler/DataHandler.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/PcdHandler.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/CvoPointCloud.hpp"
#include "utils/PointCloudLoader.hpp"

namespace cvo {

enum class RunnerDatasetType {
    PcdSequence,
    KittiLidar,
    TartanAirRgbd,
};

inline RunnerDatasetType parse_runner_dataset_type(const std::string& dataset_type) {
    if (dataset_type == "pcd" || dataset_type == "pcd_sequence") {
        return RunnerDatasetType::PcdSequence;
    }
    if (dataset_type == "kitti" || dataset_type == "kitti_lidar") {
        return RunnerDatasetType::KittiLidar;
    }
    if (dataset_type == "tartan" || dataset_type == "tartan_air" || dataset_type == "tartan_rgbd") {
        return RunnerDatasetType::TartanAirRgbd;
    }
    throw std::runtime_error("Unsupported dataset type: " + dataset_type);
}

struct RgbdCalibration {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float scale = 1.0f;
};

inline RgbdCalibration load_rgbd_calibration(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("Failed to open calibration file: " + path.string());
    }

    RgbdCalibration calib;
    in >> calib.fx >> calib.fy >> calib.cx >> calib.cy >> calib.scale;
    if (!in.good()) {
        throw std::runtime_error("Failed to parse RGB-D calibration file: " + path.string());
    }
    return calib;
}

inline pcl::PointCloud<pcl::PointXYZRGB> rgbd_to_point_cloud(const cv::Mat& rgb,
                                                             const cv::Mat& depth,
                                                             const RgbdCalibration& calib,
                                                             float max_depth = std::numeric_limits<float>::max()) {
    if (rgb.empty() || depth.empty()) {
        throw std::runtime_error("RGB-D inputs are empty");
    }
    if (depth.type() != CV_16UC1) {
        throw std::runtime_error("Expected 16-bit depth image");
    }
    if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
        throw std::runtime_error("RGB/depth image size mismatch");
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.reserve(static_cast<size_t>(rgb.rows) * static_cast<size_t>(rgb.cols));
    for (int v = 0; v < depth.rows; ++v) {
        for (int u = 0; u < depth.cols; ++u) {
            const std::uint16_t raw_depth = depth.at<std::uint16_t>(v, u);
            if (raw_depth == 0) {
                continue;
            }
            const float z = static_cast<float>(raw_depth) / calib.scale;
            if (!std::isfinite(z) || z <= 0.0f || z > max_depth) {
                continue;
            }
            pcl::PointXYZRGB pt;
            pt.z = z;
            pt.x = (static_cast<float>(u) - calib.cx) * z / calib.fx;
            pt.y = (static_cast<float>(v) - calib.cy) * z / calib.fy;
            const cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);
            pt.b = color[0];
            pt.g = color[1];
            pt.r = color[2];
            cloud.push_back(pt);
        }
    }
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;
    return cloud;
}

inline std::unique_ptr<DatasetHandler> create_dataset_handler(const std::string& dataset_type,
                                                              const std::filesystem::path& dataset_root) {
    switch (parse_runner_dataset_type(dataset_type)) {
        case RunnerDatasetType::PcdSequence:
            return std::make_unique<PcdHandler>(dataset_root.string());
        case RunnerDatasetType::KittiLidar:
            return std::make_unique<KittiHandler>(dataset_root.string(), KittiHandler::DataType::LIDAR);
        case RunnerDatasetType::TartanAirRgbd:
            return std::make_unique<TartanAirHandler>(dataset_root.string());
    }
    throw std::runtime_error("Unhandled dataset type");
}

template <typename PointT>
inline void maybe_voxel_downsample_pcl(pcl::PointCloud<PointT>& cloud, float voxel_size) {
    if (!(voxel_size > 0.0f) || cloud.empty()) {
        return;
    }
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud.makeShared());
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    pcl::PointCloud<PointT> downsampled;
    voxel.filter(downsampled);
    cloud = std::move(downsampled);
}

inline std::string voxel_cache_tag(float voxel_size) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << voxel_size;
    std::string tag = oss.str();
    for (char& ch : tag) {
        if (ch == '.') {
            ch = 'p';
        }
    }
    return tag;
}

inline std::filesystem::path kitti_downsample_cache_path(const std::filesystem::path& dataset_root,
                                                         int frame_index,
                                                         float voxel_size) {
    return dataset_root / ".cvo_cache" / ("voxel_" + voxel_cache_tag(voxel_size)) /
           ("frame_" + std::to_string(frame_index) + ".pcd");
}

template <typename PointT>
inline void load_cloud_from_dataset_handler(DatasetHandler& handler,
                                            const std::string& dataset_type,
                                            int frame_index,
                                            CvoPointCloud<PointT>& cloud,
                                            const std::filesystem::path& calibration_file = {},
                                            float tartan_max_depth = std::numeric_limits<float>::max(),
                                            float voxel_size = 0.0f) {
    handler.set_start_index(frame_index);
    switch (parse_runner_dataset_type(dataset_type)) {
        case RunnerDatasetType::PcdSequence: {
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
            if (handler.read_next_pcd(pc) != 0) {
                throw std::runtime_error("Failed to read PCD frame " + std::to_string(frame_index));
            }
            maybe_voxel_downsample_pcl(*pc, voxel_size);
            cloud = CvoPointCloud<PointT>(*pc);
            return;
        }
        case RunnerDatasetType::KittiLidar: {
            pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
            const std::filesystem::path dataset_root =
                dynamic_cast<KittiHandler&>(handler).get_data_folder();
            const bool use_cache = voxel_size > 0.0f;
            const std::filesystem::path cache_path =
                kitti_downsample_cache_path(dataset_root, frame_index, voxel_size);
            if (use_cache && std::filesystem::exists(cache_path)) {
                if (point_cloud_io::load_point_cloud(cache_path, *pc) != 0) {
                    throw std::runtime_error("Failed to load KITTI cache: " + cache_path.string());
                }
                cloud = CvoPointCloud<PointT>(*pc);
                return;
            }
            if (handler.read_next_lidar(pc) != 0) {
                throw std::runtime_error("Failed to read KITTI lidar frame " + std::to_string(frame_index));
            }
            maybe_voxel_downsample_pcl(*pc, voxel_size);
            if (use_cache) {
                std::filesystem::create_directories(cache_path.parent_path());
                point_cloud_io::save_point_cloud(cache_path, *pc);
            }
            cloud = CvoPointCloud<PointT>(*pc);
            return;
        }
        case RunnerDatasetType::TartanAirRgbd: {
            if (calibration_file.empty()) {
                throw std::runtime_error("TartanAir RGB-D loading requires a calibration file path");
            }
            cv::Mat rgb;
            cv::Mat depth;
            if (handler.read_next_rgbd(rgb, depth) != 0) {
                throw std::runtime_error("Failed to read TartanAir RGB-D frame " + std::to_string(frame_index));
            }
            const auto calib = load_rgbd_calibration(calibration_file);
            auto pcl_cloud = rgbd_to_point_cloud(rgb, depth, calib, tartan_max_depth);
            maybe_voxel_downsample_pcl(pcl_cloud, voxel_size);
            cloud = CvoPointCloud<PointT>(pcl_cloud);
            return;
        }
    }
    throw std::runtime_error("Unhandled dataset type");
}

}  // namespace cvo
