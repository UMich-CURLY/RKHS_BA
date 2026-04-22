// src/cvo/main_multiframe_bunny.cpp
#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include "cvo/CvoGPU.cuh"
#include "cvo/CvoFrameGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/PoseGraphOptimization.hpp"

using PointType = pcl::PointSemantic<3, 19>;
using CvoCloud = cvo::CvoPointCloud<PointType>;
using CvoFrame = cvo::CvoFrameGPU<PointType>;

void gen_random_poses(std::vector<Eigen::Matrix4f>& poses, int num_poses, float max_angle_axis) {
    poses.resize(num_poses);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dist(0, max_angle_axis * M_PI);
    std::uniform_real_distribution<float> trans_dist(-1.0f, 1.0f);
    poses[0] = Eigen::Matrix4f::Identity();
    for (int i = 1; i < num_poses; ++i) {
        Eigen::Vector3f axis(trans_dist(gen), trans_dist(gen), trans_dist(gen));
        axis.normalize();
        Eigen::Matrix3f R = Eigen::AngleAxisf(angle_dist(gen), axis).toRotationMatrix();
        Eigen::Vector3f t(trans_dist(gen), trans_dist(gen), trans_dist(gen));
        poses[i] = Eigen::Matrix4f::Identity();
        poses[i].block<3,3>(0,0) = R;
        poses[i].block<3,1>(0,3) = t;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <pcd_file> <cvo_param_file> <num_frames> <max_angle>\n";
        return 1;
    }
    std::string pcd_file = argv[1];
    std::string param_file = argv[2];
    int num_frames = std::stoi(argv[3]);
    float max_angle = std::stof(argv[4]);

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(pcd_file, *raw_cloud);
    // Downsample
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(raw_cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    pcl::PointCloud<pcl::PointXYZ> downsampled;
    sor.filter(downsampled);
    // Generate random poses and create clouds
    std::vector<Eigen::Matrix4f> gt_poses;
    gen_random_poses(gt_poses, num_frames, max_angle);
    std::vector<CvoCloud> clouds(num_frames);
    cvo::pgo::MapOfPoses initial_poses;
    cvo::pgo::VectorOfConstraints constraints;

    for (int i = 0; i < num_frames; ++i) {
        pcl::PointCloud<pcl::PointXYZ> transformed;
        pcl::transformPointCloud(downsampled, transformed, gt_poses[i]);
        clouds[i] = CvoCloud(transformed);

        // Store initial pose (identity for all, we will use ground truth for evaluation)
        Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
        initial_poses[i] = cvo::pgo::pose3d_from_eigen(I);
    }

    // Create constraints (all pairs)
    for (int i = 0; i < num_frames; ++i) {
        for (int j = i+1; j < num_frames; ++j) {
            cvo::pgo::Constraint3d con;
            con.id_begin = i;
            con.id_end = j;
            // Compute relative pose from ground truth as measurement
            Eigen::Matrix4f T_ij = gt_poses[i].inverse() * gt_poses[j];
            con.t_be = cvo::pgo::pose3d_from_eigen(T_ij.cast<double>());
            con.information = Eigen::Matrix<double,6,6>::Identity() * 1000.0; // high weight
            constraints.push_back(con);
        }
    }

    // Run multi‑frame alignment
    cvo::CvoParams params;
    cvo::read_CvoParams_yaml(param_file.c_str(), &params);
    cvo::CvoGPU<PointType> cvo(params);

    cvo::pgo::MapOfPoses optimized;
    cvo.align_multiframe(clouds, initial_poses, constraints, &optimized);

    // Evaluate
    for (int i = 0; i < num_frames; ++i) {
        Eigen::Matrix4d est = cvo::pgo::pose3d_to_eigen<double, Eigen::RowMajor>(optimized[i]);
        Eigen::Matrix4d gt = gt_poses[i].cast<double>();
        Eigen::Matrix4d err = est.inverse() * gt;
        const Eigen::Matrix3d R_err = err.block<3,3>(0,0);
        const Eigen::Vector3d t_err = err.block<3,1>(0,3);
        double dist = cvo::dist_se3(R_err, t_err);
        std::cout << "Frame " << i << " error: " << dist << std::endl;
    }
    return 0;
}
