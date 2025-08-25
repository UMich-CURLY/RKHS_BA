#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include "cvo/CvoGPU.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS.hpp"

using namespace std;


void read_graph_file(std::string &graph_file_path,
                     std::vector<int> & frame_inds,
                     std::vector<std::pair<int, int>> & edges,
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
                       std::vector<int> & frame_inds,
                       std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> & poses_all) {

  poses_all.resize(frame_inds.size());
  
  std::ifstream gt_file(gt_fname);

  std::string line;
  int line_ind = 0, curr_frame_ind = 0;
  
  while (std::getline(gt_file, line)) {
    
    if (line_ind < frame_inds[curr_frame_ind]) {
      line_ind ++;
      continue;
    }
    
    std::stringstream line_stream(line);
    std::string substr;
    double pose_v[12];
    int pose_counter = 0;
    while (std::getline(line_stream,substr, ' ')) {
      pose_v[pose_counter] = std::stod(substr);
      pose_counter++;
    }
    Eigen::Map<cvo::Mat34d_row> pose(pose_v);
    poses_all[curr_frame_ind] = pose;
    //if (curr_frame_ind == 2) {
    //  std::cout<<"read: line "<<frame_inds[curr_frame_ind]<<" pose is "<<poses_all[curr_frame_ind]<<std::endl;
    //}
    
    line_ind ++;
    curr_frame_ind++;
    //if (line_ind == frame_inds.size())
    if (curr_frame_ind == frame_inds.size())
      break;
  }


  gt_file.close();
}


void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  for (auto ptr : frames) {
    cvo::CvoPointCloud new_pc;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}

int main(int argc, char** argv) {

  omp_set_num_threads(24);

  string cvo_param_file(argv[1]);  
  cvo::CvoGPU cvo_align(cvo_param_file);
  
  std::string graph_file_name(argv[2]);
  std::vector<int> frame_inds;
  std::vector<std::pair<int, int>> edge_inds;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> gt_poses;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  
  read_graph_file(graph_file_name, frame_inds, edge_inds, tracking_poses);

  //std::string tracking_fname(argv[3]);
  //std::string gt_fname(argv[4]);
  //read_pose_file(tracking_fname, frame_inds, tracking_poses);
  //read_pose_file(gt_fname, frame_inds, gt_poses);

  std::string covisMapFile;
  if (argc > 3)
    covisMapFile = argv[3];

  // read point cloud
  std::vector<cvo::CvoFrameGPU::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::unordered_map<int, int> id_to_index;
  for (int i = 0; i<frame_inds.size(); i++) {

    int curr_frame_id = frame_inds[i];
    std::string frame_fname(std::to_string(curr_frame_id) + ".pcd" );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB> (frame_fname, *cloud);

    std::shared_ptr<cvo::CvoPointCloud> pc (new cvo::CvoPointCloud(*cloud));

    pcs.push_back(pc);

    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrameGPU(pc.get(), tracking_poses[i].data(), false));
    frames.push_back(new_frame);
    id_to_index[curr_frame_id] = i;

    std::cout<<"Load "<<frame_fname<<", "<<pc->positions().size()<<" number of points, pose ptr is "<<new_frame->pose_vec<<"\n";    
  }
  std::string f_name("before_BA.pcd");
  write_transformed_pc(frames, f_name);


  std::list<cvo::BinaryState::Ptr> edge_states;
  
  const cvo::CvoParams & params = cvo_align.get_params();  
  for (int i = 0; i < edge_inds.size(); i++) {
    int first_ind = id_to_index[edge_inds[i].first];
    int second_ind = id_to_index[edge_inds[i].second];
    //std::cout<<"first ind "<<first_ind<<", second ind "<<second_ind<<std::endl;
    std::cout<<"first frame "<<frames[first_ind]<<", second frame "<<frames[second_ind]<<std::endl;
    //std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[first_ind], frames[second_ind]);
    cvo::BinaryStateGPU::Ptr edge_state(new cvo::BinaryStateGPU(std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[first_ind]),
                                                                std::dynamic_pointer_cast<cvo::CvoFrameGPU>(frames[second_ind]),
                                                                &params,
                                                                cvo_align.get_params_gpu(),
                                                                params.multiframe_num_neighbors,
                                                                params.multiframe_ell_init
                                                                ));
    edge_states.push_back((edge_state));
    
  }

  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  const_flags[0] = true;

  auto start = std::chrono::system_clock::now();
  
  cvo::CvoBatchIRLS batch_irls_problem(frames, const_flags,
                                       edge_states, &cvo_align.get_params());
  batch_irls_problem.solve();//gt_poses, err_file);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> t_all = end - start;

  //cvo_align.align(frames, const_flags,
  //                edges, &time);

  std::cout<<"Align ends. Total time is "<<time<<std::endl;
  f_name="after_BA.pcd";
  write_transformed_pc(frames, f_name);


  return 0;
}
