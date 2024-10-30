#include <Eigen/Dense>

namespace cvo {
  /// change frame from gt frame to camera frame
  void gt_frame_to_cam_frame_tartanair(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & gt_poses_gt_frame,
                                       std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> & tgt_frame ) {
    Eigen::Matrix4d our_frame_from_gt_cam_frame;
    for (int j = 0; j < gt_poses_gt_frame.size(); j++) {
      Eigen::Matrix3d m;
      m = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX());
      Eigen::Matrix4d ned_from_us = Eigen::Matrix4d::Identity();
      ned_from_us.block<3,3>(0,0) = m.inverse();
      /// convert from NED to our coordinate  (camera) system    
      tgt_frame[j] = ned_from_us * gt_poses_gt_frame[j] * ned_from_us.inverse();
      
      /// now make the first frame identity
      if (j == 0) {
        our_frame_from_gt_cam_frame = tgt_frame[0].inverse();
        tgt_frame[0] = Eigen::Matrix4d::Identity();
      }  else {
        tgt_frame[j] = (our_frame_from_gt_cam_frame * tgt_frame[j]).eval();
      }
    }

  }

}
