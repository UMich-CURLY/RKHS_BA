#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <boost/filesystem.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

///#include "utils/pcl_utils.hpp"
//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/Calibration.hpp"
//#include "utils/CvoPointCloud.hpp"
//#include "utils/PointCloudIO.hpp"
//#include "cvo/CvoGPU.hpp"
//#include "cvo/CvoParams.hpp"
//#include "cvo/IRLS_State_CPU.hpp"
//#include "cvo/IRLS_State.hpp"
#include "utils/VoxelMap.hpp"


using namespace std;
using namespace boost::filesystem;

using PointType = pcl::PointXYZRGB;

extern template class cvo::VoxelMap<PointType>;
extern template class cvo::Voxel<PointType>;

template <typename PointT>
  void save_two_pc(const Eigen::Matrix4f & tmp_init_guess,
                   const pcl::PointCloud<PointT> & source,
                   const pcl::PointCloud<PointT> & target,
                   const std::string & name
                   ) {

    pcl::PointCloud<PointT> pc;
    pcl::transformPointCloud(target, pc, tmp_init_guess);
    pc = pc + source;
    pcl::io::savePCDFileASCII(name, pc);
  }

  
template<typename PointT>
float get_total_icp_dist(pcl::PointCloud<PointT> & pc1,
                         pcl::PointCloud<PointT> & pc2,
                         const Eigen::Matrix4f T_f1_to_f2
                         ) {

  pcl::PointCloud<PointT> pc2_transformed;
  pcl::transformPointCloud(pc2, pc2_transformed, T_f1_to_f2);
  float total = 0;
  for (int i = 0; i < pc2_transformed.size(); i++) {
    PointT & p2 = pc2_transformed[i];
    total += std::accumulate(pc1.begin(), pc1.end(), std::numeric_limits<float>::max(),
                         [&](float smallest , PointT & p1) {
                           float norm = (p2.getVector3fMap() - p1.getVector3fMap()).norm();                                 
                           return smallest < norm  ? smallest : norm;
                         });
      
  }
  return total;
    
}




float rand_rad() {
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return r * M_PI * 2;
}

Eigen::Matrix4f gen_rand_pose(float max_translation) {
  Eigen::Matrix3f rot;
  
  rot = Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitX())
    * Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitY())
    * Eigen::AngleAxisf( rand_rad(), Eigen::Vector3f::UnitZ());

  Eigen::Vector3f t;
  t << static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
    static_cast <float> (rand()) / static_cast <float> (RAND_MAX),
    static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  t = (t.normalized()).eval();
  t = (t*max_translation).eval();

  Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
  pose.block<3,3>(0,0) = rot;
  pose.block<3,1>(0,3) = t;
  return pose;
}

std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> gen_rand_init_pose(int discrete_rpy_num) {

  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_all;
  rot_all.reserve(discrete_rpy_num * discrete_rpy_num * discrete_rpy_num);
  for (int i = 0; i < discrete_rpy_num; i++) {
    for ( int j = 0; j < discrete_rpy_num; j++) {
      for (int k = 0; k < discrete_rpy_num; k++) {

        Eigen::Matrix3f rot;
        
        rot = Eigen::AngleAxisf( i / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitZ())
          * Eigen::AngleAxisf( j / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitY())
          * Eigen::AngleAxisf( k / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitZ());
        rot_all.emplace_back(rot);
        //std::cout<<"push "<<rot<<"\n";
      }
    }
  }
  return rot_all;
}



Eigen::Vector3f get_pc_mean(const pcl::PointCloud<PointType> & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.size(); k++)
//    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp + pc[k].getVector3fMap()).eval();
  p_mean_tmp = (p_mean_tmp) / pc.size();    
  return p_mean_tmp;
}

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  //cvo::KittiHandler kitti(argv[1], 0);
  std::string source_file(argv[1]);
  std::string target_file(argv[2]);
  int rot_discrete_num = std::stoi(argv[3]);
  srand ( time(NULL) );
  Eigen::Matrix4f idd = Eigen::Matrix4f::Identity();
  std::string name(                 "before_init_idd.pcd");  
  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> init_rots = gen_rand_init_pose(rot_discrete_num);
  std::cout<<"init rot size "<<init_rots.size()<<"\n"<<std::flush;
  //float ell = -1;
  //
  //	  ell = std::stof(argv[4]);
  pcl::PointCloud<PointType>::Ptr source_pcd(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr target_pcd(new pcl::PointCloud<PointType>);
  pcl::io::loadPCDFile(source_file, *source_pcd);
  std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
  //pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, 0.02, 0.03);
  //std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(target_file, *target_pcd);
  //pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_target = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(target_pcd, 0.02, 0.03);  
  std::cout<<"Read  target "<<target_pcd->size()<<" points\n";
  // std::shared_ptr<cvo::CvoPointCloud> target_tmp(new cvo::CvoPointCloud(*target_pcd));
  //std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud());
  //cvo::CvoPointCloud::transform(gen_rand_pose(1.0), *target_tmp, *target);
  pcl::transformPointCloud(*target_pcd, *target_pcd, gen_rand_pose(1.0));

  
  name = "before_init_raw.pcd";
  save_two_pc(idd,
              *source_pcd,
              *target_pcd,
              name 
              );

  /// subtract mean
  Eigen::Vector3f source_mean = get_pc_mean(*source_pcd);
  for (int i = 0 ; i < source_pcd->size(); i++)  {
    (*source_pcd)[i].getVector3fMap() = ((*source_pcd)[i].getVector3fMap() - source_mean ).eval();
    //  memcpy((*source)[i].label_distribution, (*fpfh_source)[i].histogram, sizeof(float)*33  );
  }
  Eigen::Vector3f target_mean = get_pc_mean(*target_pcd);
  for (int i = 0 ; i < target_pcd->size(); i++)  {
    (*target_pcd)[i].getVector3fMap() = ((*target_pcd)[i].getVector3fMap() - target_mean ).eval();
    //  memcpy((*target)[i].label_distribution, (*fpfh_target)[i].histogram, sizeof(float)*33  );
  }
  float dist = (source_mean - target_mean).norm();
  std::cout<<"source mean is "<<source_mean<<", target mean is "<<target_mean<<", dist is "<<dist<<std::endl;


  //Eigen::Vector3f dist_vec = source_mean - target_mean;

  name = "before_init_idd_centered.pcd";
  save_two_pc(idd,
              *source_pcd,
              *target_pcd,
              name
              );

  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaximumIterations (1000);

  auto start = std::chrono::system_clock::now();
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
  float curr_min_dist = 0;
  for (auto && init_rot : init_rots) {
    Eigen::Matrix4f tmp_init_guess =  Eigen::Matrix4f::Identity();
    tmp_init_guess.block<3,3>(0,0) = init_rot;

    Eigen::Matrix4f init_inv = tmp_init_guess.inverse();
    //float ip = cvo_align.function_angle(*source, * target, init_inv, init_param.ell_init_first_frame);
    float dist = get_total_icp_dist(*source_pcd, *target_pcd, tmp_init_guess);
    std::cout<<"Init guess\n"<<tmp_init_guess<<", dist is "<<dist<<"\n";
    name =  std::string("before_init_")+std::to_string(dist)+".pcd";
    save_two_pc(tmp_init_guess,
                *source_pcd,
                *target_pcd,
                name);
    
    if (dist < curr_min_dist) {
      curr_min_dist = dist;
      init_guess = tmp_init_guess;
    }
  }
  auto end = std::chrono::system_clock::now();
  double elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout<<"Chosen init guess is "<<init_guess<<", with dist "<<curr_min_dist<<", the init search takes "<<elapsed<< "ms\n";

  name =  std::string("before_init_chosen_.pcd");
  save_two_pc(init_guess,
              *source_pcd,
              *target_pcd,
              name);
  

  Eigen::Matrix4f result, init_guess_inv;
  init_guess_inv = init_guess.inverse();    
  printf("Start align... num_fixed is %d, num_moving is %d\n", source_pcd->size(), target_pcd->size());
  std::cout<<std::flush;

  double this_time = 0;
  //init_param.is_using_kdtree = true;
  //cvo_align.write_params(&init_param);

  icp.setInputSource (source_pcd);

  icp.setInputTarget (target_pcd);

  pcl::PointCloud<PointType> output;
  icp.align(output);

  result = icp.getFinalTransformation ();
  
  std::cout<<"Transform is "<<result <<"\n\n";
  std::string filename= "after_align.pcd";
  save_two_pc(result, *source_pcd, *target_pcd, filename);
  
  // append accum_tf_list for future initialization
  std::cout<<"Average registration time is "<<this_time<<std::endl;

  return 0;
}
