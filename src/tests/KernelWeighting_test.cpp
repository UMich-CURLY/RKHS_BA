#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "cvo/CvoFrameGPU.hpp"
#include "cvo/CvoGPU.cuh"
#include "utils/PointSemantic.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using PointCloud = cvo::CvoPointCloud<PointType>;

namespace {

PointType make_point(float feature0, int semantic_index) {
    PointType p;
    p.x = 0.0f;
    p.y = 0.0f;
    p.z = 0.0f;
    p.features[0] = feature0;
    p.features[1] = 0.0f;
    p.features[2] = 0.0f;
    for (int i = 0; i < 19; ++i) {
        p.label_distribution[i] = 0.0f;
    }
    p.label_distribution[semantic_index] = 1.0f;
    p.label = semantic_index;
    return p;
}

PointCloud make_cloud(const PointType& point) {
    PointCloud cloud;
    cloud.push_back(point);
    return cloud;
}

void to_pose_array(const Eigen::Matrix4d& T, double pose_arr[12]) {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> pose_map(pose_arr);
    pose_map = T.block<3, 4>(0, 0);
}

double multiframe_angle(cvo::BinaryStateGPU<PointType>& edge) {
    edge.frame1()->transform_pointcloud();
    edge.frame2()->transform_pointcloud();
    edge.update_inner_product();
    const auto& A = edge.get_A_cpu();
    return static_cast<double>(cvo::A_sum(const_cast<cvo::SparseKernelMat64*>(&A), edge.get_num_neighbors()));
}

void run_pairwise_feature_check() {
    cvo::CvoParams params;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_geometry = 1;
    params.is_using_intensity = 1;
    params.is_using_semantics = 0;
    params.c_sigma = 1.0f;
    params.c_ell = 0.1f;

    cvo::CvoGPU<PointType> solver(params);
    const PointCloud source = make_cloud(make_point(0.0f, 0));
    const PointCloud target_same = make_cloud(make_point(0.0f, 0));
    const PointCloud target_feature_diff = make_cloud(make_point(1.0f, 0));

    const float same = solver.function_angle(source, target_same, Eigen::Matrix4f::Identity(), 1.0f, true);
    const float diff = solver.function_angle(source, target_feature_diff, Eigen::Matrix4f::Identity(), 1.0f, true);

    std::cout << "Pairwise feature same=" << same << " diff=" << diff << std::endl;
    assert(same > 0.0f);
    assert(diff < same * 0.1f);
}

void run_pairwise_semantic_check() {
    cvo::CvoParams params;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_geometry = 1;
    params.is_using_intensity = 0;
    params.is_using_semantics = 1;
    params.s_sigma = 1.0f;
    params.s_ell = 0.1f;

    cvo::CvoGPU<PointType> solver(params);
    const PointCloud source = make_cloud(make_point(0.0f, 0));
    const PointCloud target_same = make_cloud(make_point(0.0f, 0));
    const PointCloud target_sem_diff = make_cloud(make_point(0.0f, 1));

    const float same = solver.function_angle(source, target_same, Eigen::Matrix4f::Identity(), 1.0f, true);
    const float diff = solver.function_angle(source, target_sem_diff, Eigen::Matrix4f::Identity(), 1.0f, true);

    std::cout << "Pairwise semantic same=" << same << " diff=" << diff << std::endl;
    assert(same > 0.0f);
    assert(diff < same * 0.1f);
}

void run_multiframe_feature_check() {
    cvo::CvoParams params;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_geometry = 1;
    params.is_using_intensity = 1;
    params.is_using_semantics = 0;
    params.c_sigma = 1.0f;
    params.c_ell = 0.1f;
    params.multiframe_num_neighbors = 1;
    params.multiframe_sparse_fill_backend = 0;

    cvo::CvoGPU<PointType> solver(params);
    PointCloud source = make_cloud(make_point(0.0f, 0));
    PointCloud target_same = make_cloud(make_point(0.0f, 0));
    PointCloud target_feature_diff = make_cloud(make_point(1.0f, 0));

    double pose0[12];
    double pose1[12];
    to_pose_array(Eigen::Matrix4d::Identity(), pose0);
    to_pose_array(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&source, pose0, false);
    auto frame_same = std::make_shared<cvo::CvoFrameGPU<PointType>>(&target_same, pose1, false);
    auto frame_diff = std::make_shared<cvo::CvoFrameGPU<PointType>>(&target_feature_diff, pose1, false);

    cvo::BinaryStateGPU<PointType> edge_same(frame0, frame_same, &solver.params(), solver.params_gpu(), 1, 1.0f);
    cvo::BinaryStateGPU<PointType> edge_diff(frame0, frame_diff, &solver.params(), solver.params_gpu(), 1, 1.0f);

    const double same = multiframe_angle(edge_same);
    const double diff = multiframe_angle(edge_diff);

    std::cout << "Multiframe feature same=" << same << " diff=" << diff << std::endl;
    assert(same > 0.0);
    assert(diff < same * 0.1);
}

void run_multiframe_semantic_check() {
    cvo::CvoParams params;
    params.sigma = 1.0f;
    params.sp_thres = 1e-6f;
    params.is_using_geometry = 1;
    params.is_using_intensity = 0;
    params.is_using_semantics = 1;
    params.s_sigma = 1.0f;
    params.s_ell = 0.1f;
    params.multiframe_num_neighbors = 1;
    params.multiframe_sparse_fill_backend = 0;

    cvo::CvoGPU<PointType> solver(params);
    PointCloud source = make_cloud(make_point(0.0f, 0));
    PointCloud target_same = make_cloud(make_point(0.0f, 0));
    PointCloud target_sem_diff = make_cloud(make_point(0.0f, 1));

    double pose0[12];
    double pose1[12];
    to_pose_array(Eigen::Matrix4d::Identity(), pose0);
    to_pose_array(Eigen::Matrix4d::Identity(), pose1);

    auto frame0 = std::make_shared<cvo::CvoFrameGPU<PointType>>(&source, pose0, false);
    auto frame_same = std::make_shared<cvo::CvoFrameGPU<PointType>>(&target_same, pose1, false);
    auto frame_diff = std::make_shared<cvo::CvoFrameGPU<PointType>>(&target_sem_diff, pose1, false);

    cvo::BinaryStateGPU<PointType> edge_same(frame0, frame_same, &solver.params(), solver.params_gpu(), 1, 1.0f);
    cvo::BinaryStateGPU<PointType> edge_diff(frame0, frame_diff, &solver.params(), solver.params_gpu(), 1, 1.0f);

    const double same = multiframe_angle(edge_same);
    const double diff = multiframe_angle(edge_diff);

    std::cout << "Multiframe semantic same=" << same << " diff=" << diff << std::endl;
    assert(same > 0.0);
    assert(diff < same * 0.1);
}

}  // namespace

int main() {
    run_pairwise_feature_check();
    run_pairwise_semantic_check();
    run_multiframe_feature_check();
    run_multiframe_semantic_check();
    return 0;
}
