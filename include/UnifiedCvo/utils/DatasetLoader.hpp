#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "utils/CvoPointCloud.hpp"
#include "utils/PointCloudLoader.hpp"

namespace cvo {

enum class DatasetType {
    PcdSequence,
    KittiLidar,
    TartanAirRgbd,
};

inline DatasetType parse_dataset_type(const std::string& dataset_type) {
    if (dataset_type == "pcd" || dataset_type == "pcd_sequence") {
        return DatasetType::PcdSequence;
    }
    if (dataset_type == "kitti" || dataset_type == "kitti_lidar") {
        return DatasetType::KittiLidar;
    }
    if (dataset_type == "tartan" || dataset_type == "tartan_air" || dataset_type == "tartan_rgbd") {
        return DatasetType::TartanAirRgbd;
    }
    throw std::runtime_error("Unsupported dataset type: " + dataset_type);
}

struct DatasetLoaderOptions {
    DatasetType dataset_type = DatasetType::PcdSequence;
    std::filesystem::path root;
    std::filesystem::path calibration_file;
    std::string tartan_depth_folder = "deep_depth";
    float tartan_max_depth = std::numeric_limits<float>::max();
};

namespace detail {

struct RgbdCalibration {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float scale = 1.0f;
};

inline std::vector<std::filesystem::path> list_files_sorted(
    const std::filesystem::path& folder,
    const std::vector<std::string>& extensions) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(folder)) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string ext = point_cloud_io::lowercase_extension(entry.path());
        if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

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

inline pcl::PointCloud<pcl::PointXYZI> load_kitti_bin_cloud(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error("Failed to open KITTI lidar file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size_bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size_bytes % (4 * static_cast<std::streamoff>(sizeof(float))) != 0) {
        throw std::runtime_error("Invalid KITTI lidar file size: " + path.string());
    }

    const size_t num_points = static_cast<size_t>(size_bytes) / (4 * sizeof(float));
    std::vector<std::array<float, 4>> raw_points(num_points);
    in.read(reinterpret_cast<char*>(raw_points.data()), size_bytes);
    if (!in.good()) {
        throw std::runtime_error("Failed to read KITTI lidar file: " + path.string());
    }

    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(num_points);
    for (const auto& raw : raw_points) {
        pcl::PointXYZI pt;
        pt.x = raw[0];
        pt.y = raw[1];
        pt.z = raw[2];
        pt.intensity = raw[3];
        cloud.push_back(pt);
    }
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;
    return cloud;
}

inline pcl::PointCloud<pcl::PointXYZRGB> load_tartan_rgbd_cloud(
    const std::filesystem::path& image_path,
    const std::filesystem::path& depth_path,
    const RgbdCalibration& calib,
    float max_depth) {
    const cv::Mat rgb = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    const cv::Mat depth = cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
    if (rgb.empty() || depth.empty()) {
        throw std::runtime_error(
            "Failed to load TartanAir frame: " + image_path.string() + " / " + depth_path.string());
    }
    if (depth.type() != CV_16UC1) {
        throw std::runtime_error("Expected 16-bit depth PNG for TartanAir frame: " + depth_path.string());
    }
    if (rgb.rows != depth.rows || rgb.cols != depth.cols) {
        throw std::runtime_error("RGB/depth size mismatch for TartanAir frame: " + image_path.string());
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

}  // namespace detail

template <typename PointT>
class DatasetLoader {
public:
    explicit DatasetLoader(DatasetLoaderOptions options)
        : options_(std::move(options)) {
        initialize();
    }

    int total_frames() const {
        return static_cast<int>(frame_labels_.size());
    }

    const std::vector<std::string>& frame_labels() const {
        return frame_labels_;
    }

    void load_frame(int frame_index, CvoPointCloud<PointT>& cloud) const {
        if (frame_index < 0 || frame_index >= total_frames()) {
            throw std::out_of_range("Frame index out of range");
        }

        switch (options_.dataset_type) {
            case DatasetType::PcdSequence: {
                point_cloud_io::load_point_cloud(frame_paths_[frame_index], cloud);
                return;
            }
            case DatasetType::KittiLidar: {
                const auto pcl_cloud = detail::load_kitti_bin_cloud(frame_paths_[frame_index]);
                cloud = CvoPointCloud<PointT>(pcl_cloud);
                return;
            }
            case DatasetType::TartanAirRgbd: {
                const auto pcl_cloud = detail::load_tartan_rgbd_cloud(
                    frame_paths_[frame_index], paired_paths_[frame_index], tartan_calib_, options_.tartan_max_depth);
                cloud = CvoPointCloud<PointT>(pcl_cloud);
                return;
            }
        }
        throw std::runtime_error("Unhandled dataset type");
    }

private:
    void initialize() {
        switch (options_.dataset_type) {
            case DatasetType::PcdSequence:
                initialize_pcd();
                break;
            case DatasetType::KittiLidar:
                initialize_kitti();
                break;
            case DatasetType::TartanAirRgbd:
                initialize_tartan();
                break;
        }
    }

    void initialize_pcd() {
        frame_paths_ = detail::list_files_sorted(options_.root, {".pcd", ".ply"});
        frame_labels_.reserve(frame_paths_.size());
        for (const auto& path : frame_paths_) {
            frame_labels_.push_back(path.stem().string());
        }
    }

    void initialize_kitti() {
        const std::filesystem::path velodyne_dir =
            std::filesystem::exists(options_.root / "velodyne") ? options_.root / "velodyne" : options_.root;
        frame_paths_ = detail::list_files_sorted(velodyne_dir, {".bin"});
        frame_labels_.reserve(frame_paths_.size());
        for (const auto& path : frame_paths_) {
            frame_labels_.push_back(path.stem().string());
        }
    }

    void initialize_tartan() {
        const std::filesystem::path image_dir = options_.root / "image_left";
        const std::filesystem::path depth_dir = options_.root / options_.tartan_depth_folder;
        frame_paths_ = detail::list_files_sorted(image_dir, {".png"});
        paired_paths_.reserve(frame_paths_.size());
        frame_labels_.reserve(frame_paths_.size());

        const std::filesystem::path calib_path =
            options_.calibration_file.empty() ? (options_.root / "cvo_calib_deep_depth.txt") : options_.calibration_file;
        tartan_calib_ = detail::load_rgbd_calibration(calib_path);

        for (const auto& image_path : frame_paths_) {
            const std::string stem = image_path.stem().string();
            frame_labels_.push_back(stem);
            const std::string prefix = stem.substr(0, stem.find("_left"));
            paired_paths_.push_back(depth_dir / (prefix + "_left_depth.png"));
        }
    }

    DatasetLoaderOptions options_;
    std::vector<std::filesystem::path> frame_paths_;
    std::vector<std::filesystem::path> paired_paths_;
    std::vector<std::string> frame_labels_;
    detail::RgbdCalibration tartan_calib_;
};

}  // namespace cvo
