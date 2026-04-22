#pragma once

#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include "CvoFrame.hpp"

namespace cvo {

template <typename PointT>
struct DevicePointCloud {
    thrust::device_vector<PointT> points;

    DevicePointCloud() = default;
    explicit DevicePointCloud(size_t size) : points(size) {}

    size_t size() const { return points.size(); }
};

template <typename PointT>
__global__ void transform_pointcloud_frame_kernel(const PointT* input,
                                                  PointT* output,
                                                  int num_points,
                                                  const float* pose_vec) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    PointT p = input[i];
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    p.x = pose_vec[0] * x + pose_vec[1] * y + pose_vec[2] * z + pose_vec[3];
    p.y = pose_vec[4] * x + pose_vec[5] * y + pose_vec[6] * z + pose_vec[7];
    p.z = pose_vec[8] * x + pose_vec[9] * y + pose_vec[10] * z + pose_vec[11];
    output[i] = p;
}

template <typename PointT>
class CvoFrameGPU : public CvoFrame<PointT> {
public:
    using Base = CvoFrame<PointT>;
    using Ptr = std::shared_ptr<CvoFrameGPU<PointT>>;
    using PointCloud = typename Base::PointCloud;
    using DeviceCloud = DevicePointCloud<PointT>;

    CvoFrameGPU(const PointCloud* pts,
                const double poses[12],
                bool is_using_kdtree = false);

    ~CvoFrameGPU();

    void transform_pointcloud() override;
    void sync_pose_to_gpu();
    void ensure_gpu_clouds();
    void release_gpu_clouds();
    bool gpu_clouds_allocated() const;

    std::shared_ptr<DeviceCloud> points_init_gpu() {
        ensure_gpu_clouds();
        return points_init_gpu_;
    }
    std::shared_ptr<DeviceCloud> points_transformed_gpu() {
        ensure_gpu_clouds();
        return points_transformed_gpu_;
    }

    size_t size() const { return this->points->size(); }
    const float* pose_vec_gpu() const { return pose_vec_gpu_; }

private:
    std::shared_ptr<DeviceCloud> points_init_gpu_;
    std::shared_ptr<DeviceCloud> points_transformed_gpu_;
    float* pose_vec_gpu_;
};

template <typename PointT>
CvoFrameGPU<PointT>::CvoFrameGPU(const PointCloud* pts,
                                 const double poses[12],
                                 bool is_using_kdtree)
    : Base(pts, poses, is_using_kdtree),
      points_init_gpu_(nullptr),
      points_transformed_gpu_(nullptr),
      pose_vec_gpu_(nullptr) {
    cudaMalloc(reinterpret_cast<void**>(&pose_vec_gpu_), sizeof(float) * 12);

    float pose_float[12];
    for (int i = 0; i < 12; ++i) pose_float[i] = static_cast<float>(poses[i]);
    cudaMemcpy(pose_vec_gpu_, pose_float, sizeof(float) * 12, cudaMemcpyHostToDevice);
}

template <typename PointT>
CvoFrameGPU<PointT>::~CvoFrameGPU() {
    cudaFree(pose_vec_gpu_);
}

template <typename PointT>
void CvoFrameGPU<PointT>::sync_pose_to_gpu() {
    float pose_float[12];
    for (int i = 0; i < 12; ++i) pose_float[i] = static_cast<float>(this->pose_vec[i]);
    cudaMemcpy(pose_vec_gpu_, pose_float, sizeof(float) * 12, cudaMemcpyHostToDevice);
}

template <typename PointT>
void CvoFrameGPU<PointT>::ensure_gpu_clouds() {
    if (points_init_gpu_ && points_transformed_gpu_) {
        return;
    }
    points_init_gpu_ = std::make_shared<DeviceCloud>(this->points->size());
    points_transformed_gpu_ = std::make_shared<DeviceCloud>(this->points->size());
    points_init_gpu_->points.assign(this->points->points().begin(), this->points->points().end());
    points_transformed_gpu_->points.assign(this->points->points().begin(), this->points->points().end());
}

template <typename PointT>
void CvoFrameGPU<PointT>::release_gpu_clouds() {
    if (points_init_gpu_) {
        points_init_gpu_->points.clear();
        points_init_gpu_->points.shrink_to_fit();
    }
    if (points_transformed_gpu_) {
        points_transformed_gpu_->points.clear();
        points_transformed_gpu_->points.shrink_to_fit();
    }
    points_init_gpu_.reset();
    points_transformed_gpu_.reset();
}

template <typename PointT>
bool CvoFrameGPU<PointT>::gpu_clouds_allocated() const {
    return static_cast<bool>(points_init_gpu_) && static_cast<bool>(points_transformed_gpu_);
}

template <typename PointT>
void CvoFrameGPU<PointT>::transform_pointcloud() {
    ensure_gpu_clouds();
    sync_pose_to_gpu();
    const int threads = 256;
    const int blocks = static_cast<int>((size() + threads - 1) / threads);
    transform_pointcloud_frame_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(points_init_gpu_->points.data()),
        thrust::raw_pointer_cast(points_transformed_gpu_->points.data()),
        static_cast<int>(size()),
        pose_vec_gpu_);
    cudaDeviceSynchronize();
}

} // namespace cvo
