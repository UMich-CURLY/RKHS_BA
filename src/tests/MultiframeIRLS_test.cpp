#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "cvo/CvoGPU.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/PointSemantic.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using PointCloud = cvo::CvoPointCloud<PointType>;

namespace {

PointCloud createAsymmetricCloud() {
    PointCloud cloud;
    const std::vector<Eigen::Vector3f> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.2f, -0.1f},
        {0.3f, 1.4f, 0.5f},
        {-0.8f, 0.4f, 1.1f},
        {0.5f, -1.2f, 0.7f},
        {1.6f, 0.8f, 1.3f},
        {-1.1f, -0.6f, 0.2f}
    };

    for (const auto& xyz : points) {
        PointType p;
        p.x = xyz.x();
        p.y = xyz.y();
        p.z = xyz.z();
        cloud.push_back(p);
    }
    return cloud;
}

PointCloud transformCloud(const PointCloud& input, const Eigen::Matrix4f& T) {
    PointCloud output;
    output.reserve(input.size());
    for (const auto& point : input.points()) {
        Eigen::Vector4f ph(point.x, point.y, point.z, 1.0f);
        Eigen::Vector3f tp = (T * ph).head<3>();
        PointType p = point;
        p.x = tp.x();
        p.y = tp.y();
        p.z = tp.z();
        output.push_back(p);
    }
    return output;
}

void toPoseArray(const Eigen::Matrix4d& T, double pose_arr[12]) {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> pose_map(pose_arr);
    pose_map = T.block<3,4>(0,0);
}

double poseDistance(const double pose_arr[12], const Eigen::Matrix4d& T_gt) {
    Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> pose_map(pose_arr);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,4>(0,0) = pose_map;
    const Eigen::Matrix4d err = T * T_gt.inverse();
    const Eigen::Matrix3d R_err = err.block<3,3>(0,0);
    const Eigen::Vector3d t_err = err.block<3,1>(0,3);
    return cvo::dist_se3<double>(R_err, t_err);
}

} // namespace

int main() {
    PointCloud cloud_ref = createAsymmetricCloud();

    Eigen::Matrix4f T_gt = Eigen::Matrix4f::Identity();
    T_gt.block<3,3>(0,0) = Eigen::AngleAxisf(0.18f, Eigen::Vector3f(0.3f, 0.7f, 0.2f).normalized()).toRotationMatrix();
    T_gt.block<3,1>(0,3) = Eigen::Vector3f(0.35f, -0.22f, 0.18f);

    PointCloud cloud_moving = transformCloud(cloud_ref, T_gt.inverse());

    cvo::CvoParams params;
    params.multiframe_max_iters = 6;
    params.multiframe_iterations_per_ell = 4;
    params.multiframe_ell_init = 1.0f;
    params.multiframe_ell_min = 0.2f;
    params.multiframe_ell_decay_rate = 0.8f;
    params.multiframe_num_neighbors = 16;
    params.multiframe_min_nonzeros = 1;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;

    cvo::CvoGPU<PointType> solver(params);

    double pose0[12];
    double pose1[12];
    toPoseArray(Eigen::Matrix4d::Identity(), pose0);
    toPoseArray(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&cloud_ref, pose0, false);
    auto frame1 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&cloud_moving, pose1, false);

    std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>> frames = {frame0, frame1};
    auto edge = std::make_shared<cvo::BinaryStateGPU<PointType>>(
        frame0, frame1, &solver.params(), solver.params_gpu(),
        params.multiframe_num_neighbors, params.multiframe_ell_init);
    std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>> edges = {edge};
    std::vector<bool> fixed_flags = {true, false};

    const double before = poseDistance(frame1->pose_vec, T_gt.cast<double>());
    const int ret = solver.align_multiframe(frames, edges, fixed_flags, nullptr);
    const double after = poseDistance(frame1->pose_vec, T_gt.cast<double>());

    std::cout << "Before: " << before << "\nAfter: " << after << std::endl;

    assert(ret == 0);
    assert(after < before);
    assert(after < 0.75);

    return 0;
}
