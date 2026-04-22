#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>
#include <boost/filesystem.hpp>

///#include "utils/pcl_utils.hpp"
//#include "dataset_handler/KittiHandler.hpp"
#include "utils/ImageStereo.hpp"
#include "utils/ImageDownsampler.hpp"
#include "utils/Calibration.hpp"
#include "utils/CvoPointCloud.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/VoxelMap.hpp"
#include "argparse/argparse.hpp"
#include "utils/PointCloudIO.hpp"
#include "utils/geometric_filter.hpp"
using namespace std;
using namespace boost::filesystem;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;





  /*
  template <typename PointT>
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr calculate_fpfh_pcl(typename pcl::PointCloud< PointT>::Ptr cloud,
                                                                float normal_radius,
                                                                float fpfh_radius){

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());

    // Create the normal estimation class, and pass the input dataset to it
    typename pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename pcl::search::KdTree<PointT>::Ptr tree_normal (new typename pcl::search::KdTree<PointT> ());
    ne.setSearchMethod (tree_normal);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (normal_radius);

    // Compute the features
    ne.compute (*cloud_normals);


    // Create the FPFH estimation class, and pass the input dataset+normals to it

    typename  pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;

    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (cloud_normals);

    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typename  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod (tree);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!

    fpfh.setRadiusSearch (fpfh_radius);

    // Compute the features
    fpfh.compute (*fpfhs);

    return fpfhs;
  }


  template <typename PointT>
  std::shared_ptr<cvo::CvoPointCloud> load_pcd_and_fpfh(const std::string & source_file,
                                                        float radius_normal,
                                                        float radius_fpfh) {
    
    typename pcl::PointCloud<PointT>::Ptr source_pcd(new typename pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(source_file, *source_pcd);
    //std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, radius_normal, radius_fpfh);
    std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));
    for (int i = 0 ; i < source->size(); i++)  {
      memcpy((*source)[i].label_distribution, (*fpfh_source)[i].histogram, sizeof(float)*33  );
    }
    return source;
    
  }

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
 

Eigen::Vector3f get_pc_mean(const cvo::CvoPointCloud & pc) {
  Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
  for (int k = 0; k < pc.num_points(); k++)
//    p_mean_tmp = (p_mean_tmp + pc.positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp + pc.at(k)).eval();
  p_mean_tmp = (p_mean_tmp) / pc.num_points();    
  return p_mean_tmp;
}
  */
void save_two_cvo_pc(const Eigen::Matrix4f & tmp_init_guess,
                     std::shared_ptr<cvo::CvoPointCloud> source,
                     std::shared_ptr<cvo::CvoPointCloud> target,
                     std::string & name
                     ) {
  cvo::CvoPointCloud  old_pc(3, 19);
  cvo::CvoPointCloud::transform(tmp_init_guess, * target, old_pc);
  cvo::CvoPointCloud sum_old = old_pc + *source;
  pcl::PointCloud<pcl::PointXYZRGB> pcd_old;  
  sum_old.export_to_pcd(pcd_old);
  std::string fname = name; //("before_align")+std::to_string(ip)+".pcd";
  
  pcl::io::savePCDFileASCII(fname, pcd_old);
  
}


int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  //cvo::KittiHandler kitti(argv[1], 0);
  // Extract values (variable names match original)
  argparse::ArgumentParser program("cvo_align_two");
  
  program.add_argument("--data_type").help("Type of data: [tartan_rgbd | tartan_test | tartan_stereo]");
  program.add_argument("--dataset_path").help("Path to input data directory");
  program.add_argument("--source_index").help("source frame index").scan<'i', int>();  
  program.add_argument("--target_index").help("target frame index").scan<'i', int>();
  program.add_argument("--cvo_param_file").help("cvo param file");
  program.add_argument("--sky_index").help("sky index").scan<'i', int>();
  program.add_argument("--is_depth_filtering").help(" whether use cvo association").scan<'i', int>();
  program.add_argument("--depth_normal_ell").help(" depth normal ell in depth_filter").scan<'g', double>();
  program.add_argument("--depth_dir_ell").help(" depth dir ell in depth_filter").scan<'g', double>();


  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  
  std::string data_type = program.get<std::string>("--data_type");
  std::string dataset_path = program.get<std::string>("--dataset_path");
  int source_index = program.get<int>("--source_index");
  int target_index = program.get<int>("--target_index");
  std::string cvo_param_file = program.get<std::string>("cvo_param_file");
  int sky_index = program.get<int>("--sky_index");
  int         is_depth_filtering                   = program.get<int>("--is_depth_filtering");
  double depth_normal_ell = program.get<double>("--depth_normal_ell");
  double depth_dir_ell = program.get<double>("--depth_dir_ell");

  


  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  cvo::TartanAirHandler tartan(dataset_path, "deep_depth");
  //tartan.set_depth_folder_name("deep_depth");
  int total_iters = tartan.get_total_number();
  std::cout<<"total num : "<<total_iters<<"\n";
  
  string calib_file;
  calib_file = dataset_path + "/cvo_calib_deep_depth.txt";
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);


  srand ( time(NULL) );
  Eigen::Matrix4f idd = Eigen::Matrix4f::Identity();
  std::string name("before_init_idd.pcd");  

  cv::Mat target_rgb, source_rgb, target_right_rgb, source_right_rgb;
  std::vector<float> target_depth, target_semantics, source_depth, source_semantics;
  std::shared_ptr<cvo::ImageRGBD<float>> target_raw, source_raw;
  std::shared_ptr<cvo::ImageStereo> target_raw_stereo, source_raw_stereo;
    
  std::shared_ptr<cvo::CvoPointCloud> source, source_full, target, target_full, source_edge, target_edge;
  if (data_type == std::string("tartan_rgbd")) {
    if (sky_index > -1) {
      tartan.set_start_index(source_index);
        
      if (tartan.read_next_rgbd_without_sky(source_rgb, source_depth, NUM_CLASSES, source_semantics, sky_index) != 0) {
        std::cout<<"source file doesn't exist\n";
        exit(0);
      }

      tartan.set_start_index(target_index);
      if (tartan.read_next_rgbd_without_sky(target_rgb, target_depth, NUM_CLASSES, target_semantics, sky_index) != 0) {
        std::cout<<"source file doesn't exist\n";
        exit(0);
      }
    

      //std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, rgb, dep, calib,1));
      target_raw.reset(new cvo::ImageRGBD<float>(target_rgb, target_depth, NUM_CLASSES, target_semantics));
      source_raw.reset(new cvo::ImageRGBD<float>(source_rgb, source_depth, NUM_CLASSES, source_semantics));


    } else {
      tartan.set_start_index(target_index);
      tartan.read_next_rgbd(target_rgb, target_depth);
      target_raw .reset(new cvo::ImageRGBD<float>(target_rgb, target_depth));

      tartan.set_start_index(source_index);    
      tartan.read_next_rgbd(source_rgb, source_depth);
      source_raw .reset(new cvo::ImageRGBD<float>(source_rgb, source_depth));
    }
    source_edge.reset(new cvo::CvoPointCloud(*source_raw,
                                        calib
                                        //, cvo::CvoPointCloud::CV_FAST
                                        //								    ));
                                        ,cvo::CvoPointCloud::DSO_EDGES,
                                        8.0f
                                        ));

    target_edge.reset(new cvo::CvoPointCloud(*target_raw, calib,
                                        //cvo::CvoPointCloud::CV_FAST));
                                        cvo::CvoPointCloud::DSO_EDGES,
                                        8.0f));

    source_full.reset(new cvo::CvoPointCloud(*source_raw,
                                             calib
                                             //, cvo::CvoPointCloud::CV_FAST
                                             //								    ));
                                             ,cvo::CvoPointCloud::FULL,
                                             8.0f
                                             ));

    target_full.reset(new cvo::CvoPointCloud(*target_raw, calib,
                                             //cvo::CvoPointCloud::CV_FAST));
                                             cvo::CvoPointCloud::FULL,
                                             8.0f));

    source = cvo::rgbd_downsampling_single_frame(source_full, source_edge, 0.15);
    target = cvo::rgbd_downsampling_single_frame(target_full, target_edge, 0.15);


    
  
  
  } else if (data_type == std::string("tartan_stereo")) {
    tartan.set_start_index(source_index);
        
    if (tartan.read_next_stereo(source_rgb, source_right_rgb, NUM_CLASSES, source_semantics) != 0) {
      std::cout<<"source file doesn't exist\n";
      exit(0);
    }

    tartan.set_start_index(target_index);
    if (tartan.read_next_stereo(target_rgb, target_right_rgb, NUM_CLASSES, target_semantics) != 0) {
      std::cout<<"source file doesn't exist\n";
      exit(0);
    }
    

    //std::shared_ptr<cvo::Frame> target(new cvo::Frame(i+1, rgb, dep, calib,1));
    target_raw_stereo.reset(new cvo::ImageStereo(target_rgb, target_right_rgb, NUM_CLASSES, target_semantics));
    source_raw_stereo.reset(new cvo::ImageStereo(source_rgb, source_right_rgb, NUM_CLASSES, source_semantics));

    source.reset(new cvo::CvoPointCloud(*source_raw_stereo,
                                        calib
                                        //, cvo::CvoPointCloud::CV_FAST
                                        //								    ));
                                        ,cvo::CvoPointCloud::CV_FAST,
                                        nullptr,
                                        10.0f
                                        ));

    target.reset(new cvo::CvoPointCloud(*target_raw_stereo, calib,
                                        //cvo::CvoPointCloud::CV_FAST));
                                        cvo::CvoPointCloud::CV_FAST,
                                        nullptr,                                        
                                        10.0f));

    source_full.reset(new cvo::CvoPointCloud(*source_raw_stereo,
                                             calib
                                             //, cvo::CvoPointCloud::CV_FAST
                                             //								    ));
                                             ,cvo::CvoPointCloud::FULL,
                                             nullptr,
                                             10.0f
                                             ));

    target_full.reset(new cvo::CvoPointCloud(*target_raw_stereo, calib,
                                             //cvo::CvoPointCloud::CV_FAST));
                                             cvo::CvoPointCloud::FULL,
                                             nullptr,                                             
                                             10.0f));

  }


 
   
  std::cout<<"source size "<<source->size()<<"\n";
  source->write_to_color_pcd("source_color.pcd");
  target->write_to_color_pcd("target_color.pcd");
  
  /*  

  pcl::io::loadPCDFile(source_file, *source_pcd);
  std::cout<<"Read  source "<<source_pcd->size()<<" points\n";
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_source = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(source_pcd, 0.02, 0.03);
  std::shared_ptr<cvo::CvoPointCloud> source(new cvo::CvoPointCloud(*source_pcd));  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(target_file, *target_pcd);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_target = cvo::calculate_fpfh_pcl<pcl::PointXYZRGB>(target_pcd, 0.02, 0.03);  
  std::cout<<"Read  target "<<target_pcd->size()<<" points\n";
  std::shared_ptr<cvo::CvoPointCloud> target_tmp(new cvo::CvoPointCloud(*target_pcd));
  std::shared_ptr<cvo::CvoPointCloud> target(new cvo::CvoPointCloud());
  cvo::CvoPointCloud::transform(gen_rand_pose(1.0), *target_tmp, *target);
  */
  name = "before_init_raw.pcd";
  save_two_cvo_pc(idd,
                  source,
                  target,
                  name 
                  );

  cvo::CvoGPU cvo_align(cvo_param_file );
  cvo::CvoParams & init_param = cvo_align.get_params();
  //init_param.ell_init = dist; //init_param.ell_init_first_frame;
  init_param.ell_init = init_param.ell_init_first_frame ;
  init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
  init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
  init_param.is_global_angle_registration = 1;
  init_param.MAX_ITER = 10000;
  
  cvo_align.write_params(&init_param);

  std::cout<<"write ell! ell init is "<<cvo_align.get_params().ell_init<<std::endl;

  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  
  Eigen::Matrix4f init_guess_inv = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f result;
  double this_time;
  cvo_align.align(*source, *target, init_guess_inv, result, nullptr,&this_time);


  
  //cvo_align.align(*source, *target, init_guess, result);
    
  std::cout<<"Transform is "<<result <<"\n\n";

  if (is_depth_filtering) {
    std::string merged_name = "before_depth_filter.pcd";
    cvo::write_transformed_pc<pcl::PointXYZRGB>(*source, *target, result, merged_name);
    std::vector<bool> inliers_pc1, inliers_pc2;
    match_two_frame(cvo_align, *source, *target,
                    depth_normal_ell, depth_dir_ell,
                    init_guess.cast<double>(), result.cast<double>(),
                    true,
                    inliers_pc1,inliers_pc2);
    source->filter_points(inliers_pc1);
    target->filter_points(inliers_pc2);
    merged_name = "after_depth_filter.pcd";
    cvo::write_transformed_pc<pcl::PointXYZRGB>(*source, *target, result, merged_name);
  }
  
  cvo::CvoPointCloud new_pc(3, 19), old_pc(3, 19);
  cvo::CvoPointCloud::transform(init_guess, * target_full, old_pc);
  cvo::CvoPointCloud::transform(result, *target_full, new_pc);
  std::cout<<"Just finished transform\n";  
  cvo::CvoPointCloud sum_old = old_pc + *source_full;
  cvo::CvoPointCloud sum_new = new_pc  + *source_full ;
  std::cout<<"Just finished CvoPointCloud concatenation\n";
  std::cout<<"num of points before and after alignment is "<<sum_old.num_points()<<", "<<sum_new.num_points()<<"\n";
  pcl::PointCloud<pcl::PointXYZRGB> pcd_old, pcd_new;  
  sum_old.export_to_pcd(pcd_old);
  sum_new.export_to_pcd(pcd_new);
  std::cout<<"Just export to pcd\n";
  std::string fname("before_align.pcd");
  pcl::io::savePCDFileASCII(fname, pcd_old);
  fname= "after_align.pcd";
  pcl::io::savePCDFileASCII(fname, pcd_new);
  // append accum_tf_list for future initialization
  std::cout<<"Average registration time is "<<this_time<<std::endl;
  save_two_cvo_pc(result,
                  source_full,
                  target_full,
                  fname
                  );

  


  return 0;
}
