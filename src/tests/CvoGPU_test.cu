// src/cvo/CvoGPU_test.cu
#include <iostream>
#include <cassert>
#include <Eigen/Dense>
#include "cvo/CvoGPU.cuh"
#include "utils/PointSemantic.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using CvoPointCloud = cvo::CvoPointCloud<PointType>;

// Helper to create a simple point cloud (cube corners)
CvoPointCloud createCubeCloud(float size, int n_per_side = 2) {
    CvoPointCloud cloud;
    for (int i = 0; i < n_per_side; ++i)
        for (int j = 0; j < n_per_side; ++j)
            for (int k = 0; k < n_per_side; ++k) {
                PointType p;
                p.x = (i / (n_per_side-1.0f) - 0.5f) * size;
                p.y = (j / (n_per_side-1.0f) - 0.5f) * size;
                p.z = (k / (n_per_side-1.0f) - 0.5f) * size;
                cloud.push_back(p);
            }
    return cloud;
}

int main() {
    // Create two point clouds with known transform
    auto source = createCubeCloud(2.0f);
    CvoPointCloud target;
    Eigen::Matrix4f T_gt = Eigen::Matrix4f::Identity();
    T_gt.block<3,3>(0,0) = Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitY()).toRotationMatrix();
    T_gt.block<3,1>(0,3) = Eigen::Vector3f(0.2f, 0.1f, 0.0f);
    // target = T_gt * source? Wait: we need T * p_target = p_source => p_target = T^{-1} * p_source
    // To generate target, we apply inverse transform to source points.
    Eigen::Matrix4f T_inv = T_gt.inverse();
    for (const auto &p : source.points()) {
        PointType tp;
        Eigen::Vector3f pos(p.x, p.y, p.z);
        Eigen::Vector3f tpos = T_inv.block<3,3>(0,0) * pos + T_inv.block<3,1>(0,3);
        tp.x = tpos.x(); tp.y = tpos.y(); tp.z = tpos.z();
        target.push_back(tp);
    }

    // Initialize CvoGPU with a parameter file (use defaults for test)
    cvo::CvoParams params;
    params.MAX_ITER = 1; // only one iteration for test
    params.ell_init = 0.5f;
    params.sigma = 0.1f;
    params.sp_thres = 0.001f;
    params.c = 1.0f;
    params.d = 1.0f;
    params.eps = 1e-5f;
    params.eps_2 = 1e-6f;
    params.nearest_neighbors_max = 8;
    // Save to temp file

    cvo::CvoGPU<PointType> cvo(params);

    // Compute initial angle
    Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity(); // start from identity
    float angle_before = cvo.function_angle(source, target, T_init, 0.5f, true);
    std::cout << "Angle before: " << angle_before << std::endl;

    // Run one iteration
    auto result = cvo.align(source, target, T_init, false);
    float angle_after = cvo.function_angle(source, target, result.T_s2t, 0.5f, true);
    std::cout << "Angle after: " << angle_after << std::endl;

    // Check that angle increased (or at least didn't decrease significantly)
    assert(angle_after >= angle_before - 1e-3f);

    // Check that transform moved closer to ground truth
    Eigen::Matrix4f T_error = result.T_s2t * T_gt; // should be identity
    Eigen::Matrix3f R_error = T_error.block<3,3>(0,0);
    Eigen::Vector3f t_error = T_error.block<3,1>(0,3);
    double dist = cvo::dist_se3<float>(R_error, t_error);
    std::cout << "Distance to GT after one iter: " << dist << std::endl;
    assert(dist < 0.5); // should have improved

    std::cout << "Test passed." << std::endl;
    return 0;
}
