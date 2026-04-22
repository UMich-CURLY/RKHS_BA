#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "cvo/CvoGPU.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/SparseKernelMat.hpp"
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
        {-0.8f, 0.4f, 1.1f}
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

PointCloud transformCloudInverse(const PointCloud& input, const Eigen::Matrix4f& T_gt) {
    PointCloud output;
    output.reserve(input.size());
    const Eigen::Matrix4f T_inv = T_gt.inverse();
    for (const auto& point : input.points()) {
        Eigen::Vector4f ph(point.x, point.y, point.z, 1.0f);
        Eigen::Vector3f tp = (T_inv * ph).head<3>();
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

double multiframeAngle(std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>>& frames,
                       std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>>& edges) {
    for (auto& frame : frames) {
        frame->transform_pointcloud();
    }

    double angle = 0.0;
    for (auto& edge : edges) {
        edge->update_inner_product();
        const cvo::SparseKernelMat64& A = edge->get_A_cpu();
        angle += cvo::A_sum(const_cast<cvo::SparseKernelMat64*>(&A), edge->get_num_neighbors());
    }
    return angle;
}

}  // namespace

int main() {
    std::cout << "angle test start" << std::endl;
    PointCloud cloud_ref = createAsymmetricCloud();

    std::vector<Eigen::Matrix4f> T_gt(2, Eigen::Matrix4f::Identity());
    T_gt[1].block<3,3>(0,0) =
        Eigen::AngleAxisf(0.16f, Eigen::Vector3f(0.3f, 0.7f, 0.2f).normalized()).toRotationMatrix();
    T_gt[1].block<3,1>(0,3) = Eigen::Vector3f(0.20f, -0.12f, 0.08f);

    PointCloud cloud_1 = transformCloudInverse(cloud_ref, T_gt[1]);

    cvo::CvoParams params;
    params.multiframe_max_iters = 1;
    params.multiframe_iterations_per_ell = 1;
    params.multiframe_ell_init = 1.0f;
    params.multiframe_ell_min = 1.0f;
    params.multiframe_ell_decay_rate = 1.0f;
    params.multiframe_num_neighbors = 16;
    params.multiframe_min_nonzeros = 1;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_kdtree = 0;

    cvo::CvoGPU<PointType> solver(params);
    std::cout << "solver ready" << std::endl;

    double pose0[12];
    double pose1[12];
    toPoseArray(Eigen::Matrix4d::Identity(), pose0);
    toPoseArray(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&cloud_ref, pose0, false);
    auto frame1 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&cloud_1, pose1, false);

    std::vector<std::shared_ptr<cvo::CvoFrameGPU<PointType>>> frames = {frame0, frame1};
    std::vector<std::shared_ptr<cvo::BinaryStateGPU<PointType>>> edges;
    edges.push_back(std::make_shared<cvo::BinaryStateGPU<PointType>>(
        frame0, frame1, &solver.params(), solver.params_gpu(),
        params.multiframe_num_neighbors, params.multiframe_ell_init));

    std::vector<bool> fixed_flags = {true, false};

    const double angle_before = multiframeAngle(frames, edges);
    std::cout << "angle before computed" << std::endl;
    const int ret = solver.align_multiframe(frames, edges, fixed_flags, nullptr);
    std::cout << "align done" << std::endl;
    const double angle_after = multiframeAngle(frames, edges);

    std::cout << "Angle before: " << angle_before
              << "\nAngle after: " << angle_after << std::endl;

    assert(ret == 0);
    assert(std::isfinite(angle_after));
    assert(angle_after >= angle_before);
    return 0;
}
