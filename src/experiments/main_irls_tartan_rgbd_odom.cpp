#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "argparse/argparse.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/PoseLoader.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
// extern template class Foo<double>;


void construct_BA_problem(cvo::CvoGPU & cvo_align,
                          std::vector<cvo::CvoFrame::Ptr> frames,
                          std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &  gt_poses,
                          int last_frame_ind,
                          int num_BA_frames
                          ) {
  // read edges to construct graph
  int start_frame_ind = last_frame_ind - num_BA_frames + 1;
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  //std::list<cvo::BinaryState::Ptr> edge_states_cpu;
  for (int i = start_frame_ind; i < last_frame_ind; i++) {
    for (int j = i+1; j < last_frame_ind+1; j++) {

      std::cout<<"first ind "<<i<<", second ind "<<j<<std::endl;
      std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[j]);
      edges.push_back(p);
    
      const cvo::CvoParams & params = cvo_align.get_params();
      cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[i]),
                                                                  std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[j]),
                                                                  &params,
                                                                  cvo_align.get_params_gpu(),
                                                                  params.multiframe_num_neighbors,
                                                                  params.multiframe_ell_init
                                                                  ));
      edge_states.push_back((edge_state));
    }
  }    
  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;
  const_flags[1] = true;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
    gt_poses_sub(gt_poses.begin()+start_frame_ind, gt_poses.begin()+last_frame_ind+1);
  std::vector<cvo::CvoFrame::Ptr> frames_sub(frames.begin()+start_frame_ind, frames.begin()+last_frame_ind+1);
  std::cout<<"Total number of BA frames is "<<frames_sub.size()<<"\n";
  
  auto start = std::chrono::system_clock::now();
  cvo::CvoBatchIRLS batch_irls_problem(frames_sub, const_flags,
                                       edge_states, &cvo_align.get_params());
  std::string err_file = std::string("err_wrt_iters_") + std::to_string(last_frame_ind)+ ".txt";
  batch_irls_problem.solve(gt_poses_sub, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;
  // cvo_align.align(frames, const_flags,
  //               edge_states, &time);

  std::cout<<"GPU Align ends. Total time is "<<double(t_all.count()) / 1000<<" seconds."<<std::endl;
  
}


void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges,
                     // optional
                     std::vector<cvo::Mat34d_row,
                      Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all) {
  std::ifstream graph_file(graph_file_path);
  
  int num_frames, num_edges;
  graph_file>>num_frames >> num_edges;
  frame_inds.resize(num_frames);
  std::cout<<"Frame indices include ";
  for (int i = 0; i < num_frames; i++) {
    graph_file >> frame_inds[i];
    std::cout<<frame_inds[i]<<", ";
  }
  std::cout<<"\nEdges include ";

  for (int i =0; i < num_edges; i++ ) {
    std::pair<int, int> p;
    graph_file >> p.first >> p.second;
    edges.push_back(p);
    std::cout<<"("<<p.first<<", "<<p.second <<"), ";
  }
  std::cout<<"\n";
  if (graph_file.eof() == false){
    std::cout<<"poses included in the graph file\n";
    poses_all.resize(num_frames);
    for (int i = 0; i < num_frames; i++) {
      double pose_vec[12];
      for (int j = 0; j < 12; j++) {
        graph_file>>pose_vec[j];
      }
      poses_all[i]  << pose_vec[0] , pose_vec[1], pose_vec[2], pose_vec[3],
        pose_vec[4], pose_vec[5], pose_vec[6], pose_vec[7],
        pose_vec[8], pose_vec[9], pose_vec[10], pose_vec[11];
      std::cout<<"read pose["<<i<<"] as \n"<<poses_all[i]<<"\n";
    }
  }
  
  graph_file.close();  
}


void read_pose_file(std::string & gt_fname,
                    std::string & selected_pose_fname,
                    std::vector<int> & frame_inds,
                    std::vector<string> & timestamps,
                    std::vector<cvo::Mat34d_row,
                      Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all) {

  poses_all.resize(frame_inds.size());
  timestamps.resize(frame_inds.size());
  std::ifstream gt_file(gt_fname);

  std::string line;
  int line_ind = 0, curr_frame_ind = 0;

  std::string gt_file_subset(selected_pose_fname);
  ofstream outfile(gt_file_subset);
  
  while (std::getline(gt_file, line)) {
    
    if (line_ind < frame_inds[curr_frame_ind]) {
      line_ind ++;
      continue;
    }

    outfile<< line<<std::endl;
    
    std::stringstream line_stream(line);
    std::string timestamp;
    double xyz[3];
    double q[4]; // x y z w
    int pose_counter = 0;

    line_stream >> timestamp;
    std::string xyz_str[3];
    line_stream >> xyz_str[0] >> xyz_str[1] >> xyz_str[2];
    xyz[0] = std::stod(xyz_str[0]);
    xyz[1] = std::stod(xyz_str[1]);
    xyz[2] = std::stod(xyz_str[2]);
    std::string q_str[4];
    line_stream >> q_str[0] >> q_str[1] >> q_str[2] >> q_str[3];
    q[0] = stod(q_str[0]);
    q[1] = stod(q_str[1]);
    q[2] = stod(q_str[2]);
    q[3] = stod(q_str[3]);
    Eigen::Quaterniond q_eigen(q[3], q[0], q[1], q[2]);
    Sophus::SO3d quat(q_eigen);
    Eigen::Vector3d trans = Eigen::Map<Eigen::Vector3d>(xyz);
    Sophus::SE3d pose_sophus(quat, trans);

    cvo::Mat34d_row pose = pose_sophus.matrix().block<3,4>(0,0);
    
    //Eigen::Map<cvo::Mat34d_row> pose(pose_v);
    poses_all[curr_frame_ind] = pose;    
    //Eigen::Matrix<double, 4,4, Eigen::RowMajor> pose_id = Eigen::Matrix<double, 4,4, Eigen::RowMajor>::Identity();
    //poses_all[curr_frame_ind] = pose_id.block<3,4>(0,0);    
    timestamps[curr_frame_ind] = timestamp;
    //if (curr_frame_ind == 2) {
    //  std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
    //}
    
    line_ind ++;
    curr_frame_ind++;
    //if (line_ind == frame_inds.size())
    if (curr_frame_ind == frame_inds.size())
      break;
  }

  outfile.close();
  gt_file.close();
}

void write_traj_file(std::string & fname,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile << t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}

template <typename T>
void write_traj_file(std::string & fname,
                     std::vector<Eigen::Matrix<T, 4, 4>,
                     Eigen::aligned_allocator<Eigen::Matrix<T, 4, 4>>> &  poses) {

  std::ofstream outfile(fname);
  for (int i = 0; i< poses.size(); i++) {
    Eigen::Matrix<T, 4, 4> pose = poses[i];//Eigen::Matrix4f::Identity();
    Sophus::SO3<T> q(pose.block(0,0,3,3));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Matrix<T, 3, 1> t(pose.block(0,3,3,1));
    outfile << t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}


void write_traj_file(std::string & fname,
                     std::vector<std::string> & timestamps,
                     std::vector<cvo::CvoFrame::Ptr> & frames ) {
  std::ofstream outfile(fname);
  for (int i = 0; i< frames.size(); i++) {
    cvo::CvoFrame::Ptr ptr = frames[i];
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    Sophus::SO3d q(pose.block<3,3>(0,0));
    auto q_eigen = q.unit_quaternion().coeffs();
    Eigen::Vector3d t(pose.block<3,1>(0,3));
    outfile <<timestamps[i]<<" "<< t(0) <<" "<< t(1)<<" "<<t(2)<<" "
            <<q_eigen[0]<<" "<<q_eigen[1]<<" "<<q_eigen[2]<<" "<<q_eigen[3]<<std::endl;
    
  }
  outfile.close();
}

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname,
                          int start_frame_ind=0, int end_frame_ind=1000000){
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  pcl::PointCloud<pcl::PointXYZ> pc_xyz_all;

  for (int i = start_frame_ind; i <= std::min((int)frames.size(), end_frame_ind); i++) {
    auto ptr = frames[i];

    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    pcl::PointCloud<pcl::PointXYZ> pc_xyz_curr;
    new_pc.export_to_pcd(pc_curr);
    new_pc.export_to_pcd(pc_xyz_curr);

    pc_all += pc_curr;
    pc_xyz_all += pc_xyz_curr;

  }
  //pcl::io::savePCDFileASCII(fname, pc_all);
  pcl::io::savePCDFileASCII(fname, pc_xyz_all);
}

std::unique_ptr<cvo::Calibration> create_calib(const std::string & data_type, 
                                               const std::string & calib_file){
  std::unique_ptr<cvo::Calibration> calib;
  cvo::KittiHandler::DataType dtype;
  if (std::strcmp(data_type.c_str(), "tum_rgbd") == 0) {
    dataset.reset(cvo::TumHandler(dataset_path) );
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
    cvo::read_pose_file_tum_format(tracking_traj_file,
                                   start_ind,
                                   last_ind,
                                   tracking_poses);
    
  } else if (std::strcmp(data_type.c_str(), "tartan_rgbd") == 0) {
    dataset.reset(cvo::TartanAirHandler(dataset_path) );
    dataset.set_depth_folder_name("deep_depth");
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
    cvo::read_pose_file_tartan_format(tracking_traj_file,
                                      start_ind,
                                      last_ind,
                                      tracking_poses);
    
  } else if (std::strcmp(data_type.c_str(), "kitti_stereo") == 0) {
    dtype = cvo::KittiHandler::DataType::STEREO;
    dataset.reset(new cvo::KittiHandler(dataset_path, dtype, cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME));
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::STEREO));        
  } else if (std::strcmp(data_type.c_str(), "kitti_lidar") == 0) {
    dtype = cvo::KittiHandler::DataType::LIDAR;
    dataset.reset(new cvo::KittiHandler(dataset_path, dtype, cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME));    
  } else {
    ASSERT(false, "unknown data type");
  }
  return std::move(dataset);  
  
}


std::unique_ptr<cvo::DatasetHandler> create_dataset(const std::string & data_type,
                                                    const std::string & dataset_path) {

  std::unique_ptr<cvo::DatasetHandler> dataset;
  std::unique_ptr<cvo::Calibration> calib;
  cvo::KittiHandler::DataType dtype;
  if (std::strcmp(data_type.c_str(), "tum_rgbd") == 0) {
    dataset.reset(cvo::TumHandler(dataset_path) );
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
    cvo::read_pose_file_tum_format(tracking_traj_file,
                                   start_ind,
                                   last_ind,
                                   tracking_poses);
    
  } else if (std::strcmp(data_type.c_str(), "tartan_rgbd") == 0) {
    dataset.reset(cvo::TartanAirHandler(dataset_path) );
    dataset.set_depth_folder_name("deep_depth");
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::RGBD));
    cvo::read_pose_file_tartan_format(tracking_traj_file,
                                      start_ind,
                                      last_ind,
                                      tracking_poses);
    
  } else if (std::strcmp(data_type.c_str(), "kitti_stereo") == 0) {
    dtype = cvo::KittiHandler::DataType::STEREO;
    dataset.reset(new cvo::KittiHandler(dataset_path, dtype, cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME));
    calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::STEREO));        
  } else if (std::strcmp(data_type.c_str(), "kitti_lidar") == 0) {
    dtype = cvo::KittiHandler::DataType::LIDAR;
    dataset.reset(new cvo::KittiHandler(dataset_path, dtype, cvo::KittiHandler::LidarCamCalibType::LIDAR_FRAME));    
  } else {
    ASSERT(false, "unknown data type");
  }
  return std::move(dataset);  
}


argparse::ArgumentParser parser(int argc, char** argv) {

  argparse::ArgumentParser parser("irls_odom");
  parser.add_argument("--data-type").required();
  parser.add_argument("--dataset-path").required();
  parser.add_argument("--cvo-param-file").required();
  parser.add_argument("--cvo-calib-file").required();
  parser.add_argument("--num-ba-frames").default_value(4);
  parser.add_argument("--init-traj-file").default_value(std::string{""});
  parser.add_argument("--ba-traj-file").required();
  parser.add_argument("--is-edge-only").default_value(false);
  parser.add_argument("--start-ind").default_value(0).scan<'i', int>();
  parser.add_argument("--max-last-ind").default_value(10000).scan<'i',int>();
  parser.add_argument("--sky-label").default_value(-1).scan<'i', int>();

  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  return parser;
}


int main(int argc, char** argv) {

  //  omp_set_num_threads(24);
  auto parser = parse(argc, argv);

  //cvo::TartanAirHandler tartan(argv[1]);
  //dataset.set_depth_folder_name("deep_depth");
  std::string data_type(parser.get<std::string>("--data-type"));
  std::string dataset_path(parser.get<std::string>("--dataset-path"));
  string cvo_param_file(parser.get<std::string>("--cvo-param-file"));    
  string calib_file_name(parser.get<std::string>("--cvo-calib-file"));
  int num_BA_frames = std::stoi(parser.get<int>("--num-ba-frames"));
  std::string tracking_traj_file(parser.get<std::string>("--init-traj-file"));
  std::string BA_traj_file(parser.get<std::string>("--ba-traj-file"));
  int is_edge_only = std::stoi(parser.get<bool>("--is-edge-only"));
  int start_ind = std::stoi(parser.get<int>("--start-ind"));
  int max_last_ind = std::stoi(parser.get<int>("--max-last-ind"));
  int sky_label = std::stoi(parser.get<int>("--sky-label"));
  int last_ind = std::min(max_last_ind+1, dataset.get_total_number())-1;
  int total_iters = last_ind - start_ind ;  
  bool is_recording_full = false;

  cvo::CvoGPU cvo_align(cvo_param_file);
  string calib_file, gt_pose_name;
  calib_file = std::string(dataset_path) +"/" + calib_file_name;

  std::unique_ptr<cvo::DatasetHandler> dataset;
  std::unique_ptr<cvo::Calibration> calib;
  std::vector<Eigen::Matrix4d,
              Eigen::aligned_allocator<Eigen::Matrix4d>> tracking_poses(total_iters);


  
  std::vector<cvo::Mat34d_row,
              Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses(total_iters);

  
  // read point cloud
  std::map<int, cvo::CvoFrame::Ptr> frames;
  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::map<int, std::shared_ptr<cvo::CvoPointCloud>> pcs_full;
  for (int i = 0; i< tracking_poses.size(); i++) {
    std::cout<<"new frame "<<i+start_ind<<" out of "<<total_iters<<"\n";
    
    dataset->set_start_index(i+start_ind);
    cv::Mat rgb;
    vector<float> depth, semantics;
    dataset->read_next_rgbd_without_sky(rgb, depth, NUM_CLASSES, semantics, sky_label);
    std::shared_ptr<cvo::ImageRGBD<float>> raw(new cvo::ImageRGBD<float>(rgb, depth));

    
    std::shared_ptr<cvo::CvoPointCloud> pc_full;
    pc_full = std::make_shared<cvo::CvoPointCloud> (*raw,  *calib, cvo::CvoPointCloud::FULL);
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, *calib, cvo::CvoPointCloud::DSO_EDGES));

    std::cout<<"is_edge_only is "<<is_edge_only<<"\n";
    std::shared_ptr<cvo::CvoPointCloud> pc;
    float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;  
    if (is_edge_only)
      pc = downsample_points(is_edge_only, raw, pc_full, pc_edge_raw, leaf_size);
    else
      pc = pc_edge_raw;
    pcs.push_back(pc);
    pcs_full.push_back(pc_full);


    /// generate tracking poses for the latest frame
    if (i == 0) {
      tracking_poses[i] = Eigen::Matrix4f::Identity();
      auto id_row = cvo::Mat44d_row::Identity();
      BA_poses[i] = id_row.block<3,4>(0,0);
    } else if (i == 1) {
      Eigen::Matrix4f init_guess_inv = Eigen::Matrix4f::Identity();
      cvo_align.align(*pcs[i-1], *pcs[i], init_guess_inv, tracking_poses[i]);
      auto tracking_row = tracking_poses[i].cast<double>();
      BA_poses[i] = tracking_row.block<3,4>(0,0);
    } else {
      Eigen::Matrix4f init_guess_inv = tracking_poses[i-1].inverse() * tracking_poses[i-2];
      Eigen::Matrix4f tracking_result;
      cvo_align.align(*pcs[i-1], *pcs[i], init_guess_inv, tracking_result);
      tracking_poses[i] = tracking_poses[i-1] * tracking_result;

      cvo::Mat44d_row T_last_to_curr = tracking_result.cast<double>();
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row_last = cvo::Mat44d_row::Identity();
      T_row_last.block<3,4>(0,0) = BA_poses[i-1];
      BA_poses[i] = (T_row_last * T_last_to_curr).block<3,4>(0,0); 
    }

    /// construct CvoFrame 
    double * poses_data = BA_poses[i].data();
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pcs.back().get(), poses_data, cvo_align.get_params().is_using_kdtree));
    cvo::CvoFrame::Ptr new_full_frame(new cvo::CvoFrameGPU(pcs_full.back().get(), poses_data,  cvo_align.get_params().is_using_kdtree));
    frames.push_back(new_frame);
    frames_full.push_back(new_full_frame);
    

    if (i > num_BA_frames-1) {
      int start_frame = i-num_BA_frames+1;
      int end_frame = i;
      std::string f_name_full = std::string("before_BA_full_") + std::to_string(i+start_ind) + std::string(".pcd") ;
      std::string f_name = std::string("before_BA_") + std::to_string(i+start_ind) + std::string(".pcd") ;
      //write_transformed_pc(frames_full, f_name_full, start_frame, end_frame);
      write_transformed_pc(frames, f_name, start_frame, end_frame);

      /// Multiframe alignment
      construct_BA_problem(cvo_align, frames, gt_poses, i, num_BA_frames);

      std::cout<<"copy result to frames_full\n";
      for (int j = i-num_BA_frames+1; j < end_frame+1; j++) 
        memcpy(frames_full[j]->pose_vec, frames[j]->pose_vec, sizeof(double)*12);
      std::cout<<"write results to pcd\n";
      f_name = std::string("after_BA_") + std::to_string(i+start_ind) + std::string(".pcd") ;
      f_name_full = std::string("after_BA_full_") + std::to_string(i+start_ind) + std::string(".pcd") ;
      //write_transformed_pc(frames_full, f_name_full, start_frame, end_frame);
      write_transformed_pc(frames, f_name, start_frame, end_frame);

      pcs[start_frame].reset();
      pcs_full[start_frame].reset();
      dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[start_frame])->clear_points();
      dynamic_pointer_cast<cvo::CvoFrameGPU>(frames_full[start_frame])->clear_points();
    }
  }

  std::cout<<"Write traj to file\n";
  write_traj_file(BA_traj_file,frames);
  std::string gt_fname("groundtruth.txt");
  write_traj_file<double>(gt_fname,gt_poses);  
  write_traj_file<float>(tracking_traj_file, tracking_poses);
  return 0;
}
