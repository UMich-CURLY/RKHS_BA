#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <vector>
#include "cvo/CvoFrame.hpp"

namespace cvo {
  
  struct PoseErrorMetric {
    enum Type {LOG, FROBENIUS};
  };
  
  double log_err_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                              std::vector<Sophus::SE3d> & poses_new) {
    double change = 0;
    for (int i = 0; i < poses_old.size(); i++) {
      change += (poses_old[i].inverse() * poses_new[i]).log().squaredNorm();
    }
    return change;
  }

  double frobenius_err_of_all_poses(std::vector<Sophus::SE3d> & poses_old,
                                    std::vector<Sophus::SE3d> & poses_new) {
    double change = 0;
    for (int i = 0; i < poses_old.size(); i++) {
      std::cout<<"comparing \n"<<poses_old[i].matrix()<<"\n and "<<poses_new[i].matrix()<<"\n";
	    change += (poses_old[i].matrix() -  poses_new[i].matrix()).squaredNorm();
    }
    return change;
  } 

  double eval_pose(PoseErrorMetric::Type metric,
                   const std::map<int, cvo::CvoFrame::Ptr> & frames,
                   const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt_poses_all) {
    std::vector<Sophus::SE3d> poses;
    std::vector<Sophus::SE3d> gt;

  
    for (auto & [ind, frame] : frames) {
      Eigen::Matrix4d pose = frame->pose_cpu();
      Sophus::SE3d T(pose.block<3,3>(0,0), pose.block<3,1>(0,3));
      poses.push_back(T);
      Sophus::SE3d T_gt(gt_poses_all[ind].block<3,3>(0,0), gt_poses_all[ind].block<3,1>(0,3));
      gt.push_back(T_gt);
      
    }

    if (metric == PoseErrorMetric::LOG) {
      return log_err_of_all_poses(poses, gt);
    } else {
      return frobenius_err_of_all_poses(poses, gt);
    }
  }

  double eval_pose(PoseErrorMetric::Type metric,
                   const std::map<int, cvo::CvoFrame::Ptr> & frames,
                   const std::map<int, Eigen::Matrix4d 
		   		  //,std::less<int>,
		   		  //Eigen::aligned_allocator<Eigen::Matrix4d>
				  > & gt_poses_all) {
    std::vector<Sophus::SE3d> poses;
    std::vector<Sophus::SE3d> gt;

  
    for (auto & [ind, frame] : frames) {
      Eigen::Matrix4d pose = frame->pose_cpu();
      Sophus::SE3d T(pose.block<3,3>(0,0), pose.block<3,1>(0,3));
      poses.push_back(T);
      Eigen::Matrix4d pose_gt = gt_poses_all.find(ind)->second;
      Sophus::SE3d T_gt(pose_gt.block<3,3>(0,0), pose_gt.block<3,1>(0,3));
      gt.push_back(T_gt);
    }

    if (metric == PoseErrorMetric::LOG) {
      return log_err_of_all_poses(poses, gt);
    } else {
      return frobenius_err_of_all_poses(poses, gt);
    }  
  }


}
