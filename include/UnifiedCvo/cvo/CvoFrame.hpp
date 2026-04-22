#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "utils/CvoPointCloud.hpp"

namespace cvo {

template <typename PointT>
class CvoFrame {
public:
    using Ptr = std::shared_ptr<CvoFrame<PointT>>;
    using PointCloud = CvoPointCloud<PointT>;

    CvoFrame(const PointCloud* pts,
             const double poses[12],
             bool is_using_kdtree = false)
        : points(pts), is_using_kdtree_(is_using_kdtree)
    {
        memcpy(pose_vec, poses, 12 * sizeof(double));
    }

    virtual ~CvoFrame() = default;

    virtual void transform_pointcloud() {}

    const PointCloud* points;                // non‑owning pointer to point cloud
    double pose_vec[12];                     // 3x4 row‑major matrix [R t]
    bool is_using_kdtree_;

    Eigen::Matrix4d pose_cpu() const {
        Eigen::Matrix4d pose;
        pose << pose_vec[0], pose_vec[1], pose_vec[2], pose_vec[3],
                pose_vec[4], pose_vec[5], pose_vec[6], pose_vec[7],
                pose_vec[8], pose_vec[9], pose_vec[10], pose_vec[11],
                0.0, 0.0, 0.0, 1.0;
        return pose;
    }

    // ID management for multi‑frame (optional)
    void set_id(int id) { id_ = id; }
    int  get_id() const { return id_; }

protected:
    int id_ = -1;
};

} // namespace cvo
