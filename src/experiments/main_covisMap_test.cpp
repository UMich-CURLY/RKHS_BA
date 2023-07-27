#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>

#include "cvo/CvoGPU.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoFrame.hpp"
#include "utils/VoxelMap.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/ImageRGBD.hpp"
#include "utils/Calibration.hpp"

using namespace std;

extern template class cvo::VoxelMap<pcl::PointXYZRGB>;
extern template class cvo::Voxel<pcl::PointXYZRGB>;
//extern template class Foo<double>;

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

void write_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames, std::string & fname, int max_frames=-1) {
  pcl::PointCloud<pcl::PointXYZRGB> pc_all;
  int counter = 0;
  for (auto ptr : frames) {
    if (counter == max_frames) break;
    cvo::CvoPointCloud new_pc(5, 0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3,4>(0,0) = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);

    pcl::PointCloud<pcl::PointXYZRGB> pc_curr;
    new_pc.export_to_pcd(pc_curr);

    pc_all += pc_curr;
    counter++;
  }
  pcl::io::savePCDFileASCII(fname, pc_all);
}

void merge_transformed_pc(std::vector<cvo::CvoFrame::Ptr> & frames,  cvo::CvoPointCloud & pc_all) {
  std::cout<<"start merge_transformed_pc\n";
  if (frames.size() == 0)
    return;
  
  int num_points = 0;
  for (auto ptr : frames) {
    num_points += ptr->points->num_points();
  }
  //pc_all.reserve(num_points, frames[0]->points->feature_dimensions(),
  //               frames[0]->points->num_classes());
  for (auto ptr : frames) {
    
    //if (counter == max_frames) break;
    cvo::CvoPointCloud new_pc(5,0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    cvo::Mat34d_row pose_eigen = Eigen::Map<cvo::Mat34d_row>(ptr->pose_vec);
    pose.block<3,4>(0,0) = pose_eigen;
    
    Eigen::Matrix4f pose_f = pose.cast<float>();
    std::cout<<"Before transform first point is  "<<ptr->points->positions()[0]<<std::endl;
    std::cout<<"Before transform 2nd point is  "<<ptr->points->positions()[1]<<std::endl;
    
    cvo::CvoPointCloud::transform(pose_f, *ptr->points, new_pc);
    std::cout<<"after transform first point is  "<<new_pc.positions()[0]<<std::endl;
    std::cout<<"after transform 2nd point is  "<<new_pc.positions()[1]<<std::endl;
    pc_all = pc_all + new_pc;
    std::cout<<"2nd point of pc_all is "<<pc_all.positions()[1]<<std::endl;
    std::cout<<"Last point of pc_all is "<<pc_all.positions()[pc_all.num_points()-1]<<std::endl;
    std::cout<<"pc all size is "<<pc_all.num_points()<<std::endl;
  }

}


int main(int argc, char** argv) {

  omp_set_num_threads(24);

  cvo::TartanAirHandler tartan(argv[1]);
  string cvo_param_file(argv[2]);    
  std::string graph_file_name(argv[3]);


  //std::string tracking_fname(argv[4]);

  int num_const_frames = 1;
  //if (argc > 4)
  num_const_frames = std::stoi(std::string(argv[4]));
  std::string graph_file_folder;
  //if (argc > 5)
  graph_file_folder = std::string(argv[5]);

  std::string covisMapFile;
  if (argc > 6)
    covisMapFile = std::string(argv[6]);
  
  int total_iters = tartan.get_total_number();
  //vector<string> vstrRGBName = tum.get_rgb_name_list();


  cvo::CvoGPU cvo_align(cvo_param_file);
  string calib_file;
  calib_file = string(argv[1] ) +"/cvo_calib.txt"; 
  cvo::Calibration calib(calib_file, cvo::Calibration::RGBD);

  

  std::vector<int> frame_inds;
  std::vector<std::pair<int, int>> edge_inds;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> BA_poses;
  read_graph_file(graph_file_name, frame_inds, edge_inds, BA_poses);

  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> gt_poses;
  std::vector<cvo::Mat34d_row, Eigen::aligned_allocator<cvo::Mat34d_row>> tracking_poses;
  std::vector<std::string> timestamps;

  std::string tracking_subset_poses_fname("cvo_track_poses.txt");
  //read_pose_file(tracking_fname, tracking_subset_poses_fname , frame_inds, timestamps, tracking_poses);

  // read point cloud
  std::vector<cvo::CvoFrame::Ptr> frames;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs;
  std::vector<std::shared_ptr<cvo::CvoPointCloud>> pcs_full;
  std::vector<std::shared_ptr<cvo::CvoFrame>> frames_full;
  std::unordered_map<int, int> id_to_index;
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  float leaf_size = cvo_align.get_params().multiframe_downsample_voxel_size;  
  for (int i = 0; i<frame_inds.size(); i++) {

    int curr_frame_id = frame_inds[i];
    
    tartan.set_start_index(curr_frame_id);
    cv::Mat rgb;
    vector<float> depth    ;
    tartan.read_next_rgbd(rgb, depth);

    cvo::VoxelMap<pcl::PointXYZRGB> edge_voxel(leaf_size / 10); // /10
    cvo::VoxelMap<pcl::PointXYZRGB> surface_voxel(leaf_size);

    /*    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(graph_file_folder + "/" + std::to_string(curr_frame_id)+".pcd",
                         *raw_pcd);
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(*raw_pcd));
    std::cout<<"read number points is "<<pc->num_points()<<std::endl;
    */
    
    std::shared_ptr<cvo::ImageRGBD<float>> raw(new cvo::ImageRGBD<float>(rgb, depth));
    
    std::shared_ptr<cvo::CvoPointCloud> pc_full(new cvo::CvoPointCloud(*raw,  calib, cvo::CvoPointCloud::FULL));
    


    
    std::shared_ptr<cvo::CvoPointCloud> pc_edge_raw(new cvo::CvoPointCloud(*raw, calib, cvo::CvoPointCloud::DSO_EDGES));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_edge(new pcl::PointCloud<pcl::PointXYZRGB>);    
    pc_edge_raw->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_edge);
    for (int k = 0; k < raw_pcd_edge->size(); k++) {
      edge_voxel.insert_point(&raw_pcd_edge->points[k]);
    }
    std::vector<pcl::PointXYZRGB*> edge_results = edge_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZRGB> edge_pcl;
    for (int k = 0; k < edge_results.size(); k++)
      edge_pcl.push_back(*edge_results[k]);
    std::cout<<"edge voxel selected points "<<edge_pcl.size()<<std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pcd_surface(new pcl::PointCloud<pcl::PointXYZRGB>);    
    pc_full->export_to_pcd<pcl::PointXYZRGB>(*raw_pcd_surface);
    for (int k = 0; k < raw_pcd_surface->size(); k++) {
      surface_voxel.insert_point(&raw_pcd_surface->points[k]);
    }
    std::vector<pcl::PointXYZRGB*> surface_results = surface_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZRGB> surface_pcl;
    for (int k = 0; k < surface_results.size(); k++)
      surface_pcl.push_back(*surface_results[k]);
    std::cout<<"surface voxel selected points "<<surface_pcl.size()<<std::endl;
    
    
    // std::cout<<"start voxel filtering...\n"<<std::flush;
    //sor.setInputCloud (raw_pcd);

    //sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    //sor.filter (*raw_pcd);
    //std::cout<<"construct filtered cvo points with voxel size "<<leaf_size<<"\n"<<std::flush;
    std::shared_ptr<cvo::CvoPointCloud> pc_edge(new cvo::CvoPointCloud(edge_pcl, cvo::CvoPointCloud::GeometryType::EDGE));
    std::shared_ptr<cvo::CvoPointCloud> pc_surface(new cvo::CvoPointCloud(surface_pcl, cvo::CvoPointCloud::GeometryType::SURFACE));
    std::shared_ptr<cvo::CvoPointCloud> pc(new cvo::CvoPointCloud(5,0));
    *pc = *pc_edge + *pc_surface;
    
    std::cout<<"Voxel number points is "<<pc->num_points()<<std::endl;

    // pcl::PointCloud<pcl::PointXYZRGB> pcd_to_save;
    pc->write_to_color_pcd(std::to_string(curr_frame_id)+".pcd");


    
    
    std::cout<<"Load "<<curr_frame_id<<", "<<pc->positions().size()<<" number of points\n"<<std::flush;
    pcs.push_back(pc);
    pcs_full.push_back(pc_full);

    double * poses_data = nullptr;
    //if (BA_poses.size())

    Eigen::Matrix4d id_mat = Eigen::Matrix4d::Identity();
    if (BA_poses.size() == frame_inds.size())
      poses_data = BA_poses[i].data();
    else 
      poses_data = id_mat.data();
    
    cvo::CvoFrame::Ptr new_frame(new cvo::CvoFrame(pc.get(), poses_data, false));
    cvo::CvoFrame::Ptr new_full_frame(new cvo::CvoFrame(pc_full.get(), poses_data, false));
    frames.push_back(new_frame);
    frames_full.push_back(new_full_frame);
    id_to_index[curr_frame_id] = i;
  }


  
  std::string f_name_full("before_BA_full.pcd");
  std::string f_name("before_BA.pcd");
  write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);

  cvo::CvoPointCloud before_ba_cvo(5,0);
  merge_transformed_pc(frames, before_ba_cvo);
  pcl::PointCloud<pcl::PointXYZRGB> pc_to_write;
  before_ba_cvo.export_to_pcd(pc_to_write);
  std::string nfname("before_ba_cvo.pcd");
  std::cout<<"write to before_ba_cvo.pcd";
  pcl::io::savePCDFileASCII(nfname, pc_to_write);
  std::cout<<"fin\n";
  //before_ba_cvo.write_to_color_pcd("before_ba_cvo.pcd");

  // read edges to construct graph
  std::list<std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr>> edges;
  std::list<cvo::BinaryState::Ptr> edge_states;
  for (int i = 0; i < edge_inds.size(); i++) {
    int first_ind = id_to_index[edge_inds[i].first];
    int second_ind = id_to_index[edge_inds[i].second];
     std::cout<<"first ind "<<first_ind<<", second ind "<<second_ind<<std::endl;
    std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[first_ind], frames[second_ind]);
    edges.push_back(p);
    
    const cvo::CvoParams & params = cvo_align.get_params();
    cvo::BinaryState::Ptr edge_state(new cvo::BinaryStateCPU(frames[first_ind],
                                                             frames[second_ind],
                                                             &params,
                                                             params.multiframe_num_neighbors,
                                                             params.multiframe_ell_init
                                                             ));
    edge_states.push_back(edge_state);
    
  }

  /*
  if (covisMapFile.size()) {
    for (int i = 0; i < frames.size() - 1; i++) {
      std::pair<cvo::CvoFrame::Ptr, cvo::CvoFrame::Ptr> p(frames[i], frames[frames.size()-1]);
      edges.push_back(p);
      const cvo::CvoParams & params = cvo_align.get_params();
      cvo::BinaryState::Ptr edge_state(new cvo::BinaryStateCPU(frames[i],
                                                               frames[frames.size()-1],
                                                               &params,
                                                               params.multiframe_kdtree_num_neighbors,
                                                               params.multiframe_ell_init * 4
                                                             ));
      edge_states.push_back(edge_state);
      
    }
  }
  */


  /*
  f_name_full = ("before_BA_full.pcd");
  f_name = ("before_BA.pcd");
  for (int i = 0; i < frames.size(); i++) {
    double * raw =  BA_poses[i].data(); //Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(BA_poses[i]);
    memcpy(frames_full[i]->pose_vec, raw, sizeof(double)*12);
    memcpy(frames[i]->pose_vec, raw, sizeof(double)*12);
  }
  write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);

  */

  double time = 0;
  std::vector<bool> const_flags(frames.size(), false);
  std::cout<<"Const frames include ";
  if (covisMapFile.size())
    //const_flags[const_flags.size()-1] = true;
    const_flags[0] = true;
  else {
    for (int i = 0; i < num_const_frames; i++) {
      const_flags[i] = true;
      std::cout<<frame_inds[i]<<", ";
    }
    std::cout<<std::endl;
  }
  f_name="const_BA.pcd";
  f_name_full="const_BA_full.pcd";
  write_transformed_pc(frames_full, f_name_full, num_const_frames);
  write_transformed_pc(frames, f_name, num_const_frames);
  
  cvo_align.align(frames, const_flags,
                  edge_states, &time);

  std::cout<<"Align ends. Total time is "<<time<<" seconds."<<std::endl;
  f_name="after_BA.pcd";
  f_name_full="after_BA_full.pcd";
  for (int i = 0; i < frames.size(); i++) {
    memcpy(frames_full[i]->pose_vec, frames[i]->pose_vec, sizeof(double)*12);
  }
  write_transformed_pc(frames_full, f_name_full);
  write_transformed_pc(frames, f_name);



  if (covisMapFile.size() > 0) {

    std::shared_ptr<cvo::CvoPointCloud> tmpMap(new cvo::CvoPointCloud(5, 0));
    merge_transformed_pc(frames, *tmpMap);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_voxel_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    tmpMap->export_to_pcd(*tmp_voxel_pcd);
    pcl::io::savePCDFileASCII("merged_pc.pcd", *tmp_voxel_pcd);
    cvo::VoxelMap<pcl::PointXYZRGB> tmp_voxel(leaf_size / 2); // /10
    for (int k = 0; k < tmp_voxel_pcd->size(); k++) {
      auto p_ptr = &tmp_voxel_pcd->points[k];
      Eigen::Vector3f p_eigen = p_ptr->getVector3fMap();
      if(! (p_eigen(0) != p_eigen(0)
            || p_eigen(1) != p_eigen(1)
            || p_eigen(2) != p_eigen(2)
            ))
        tmp_voxel.insert_point(p_ptr);
    }
    std::vector<pcl::PointXYZRGB*> tmp_results = tmp_voxel.sample_points();
    pcl::PointCloud<pcl::PointXYZRGB> tmp_pcl;
    for (int k = 0; k < tmp_results.size(); k++) {

    
      tmp_pcl.push_back(*tmp_results[k]);
    }
    std::cout<<"tmp voxel selected points "<<tmp_pcl.size()<<std::endl;
    std::shared_ptr<cvo::CvoPointCloud> tmp_pc(new cvo::CvoPointCloud(tmp_pcl));
    tmp_pc->write_to_color_pcd("tmp_pc.pcd");

    std::cout<<"tmp Map with points "<<tmp_pc->num_points()<<std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr covis_pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(graph_file_folder + "/" + covisMapFile,
                        *covis_pcd);
    std::shared_ptr<cvo::CvoPointCloud> covis_pc(new cvo::CvoPointCloud(*covis_pcd));
    std::cout<<"covis Map with points "<<covis_pc->num_points()<<std::endl;
    
    //pcs.push_back(pc);
    cvo::CvoFrame::Ptr tmp_pc_frame(new cvo::CvoFrame(tmp_pc.get(), frames[0]->pose_vec, false));
    cvo::CvoFrame::Ptr covis_pc_frame(new cvo::CvoFrame(covis_pc.get(), frames[0]->pose_vec, false));

    std::cout<<"just constructed frames\n";
    std::vector<cvo::CvoFrame::Ptr> covisFrames;
    covisFrames.push_back(tmp_pc_frame);
    covisFrames.push_back(covis_pc_frame);
    f_name = "before_covis_BA.pcd";
    write_transformed_pc(covisFrames, f_name);

    Eigen::Vector3f p_mean_covis = Eigen::Vector3f::Zero();
    Eigen::Vector3f p_mean_tmp = Eigen::Vector3f::Zero();
    for (int k = 0; k < tmp_pc->num_points(); k++)
      p_mean_tmp = (p_mean_tmp + tmp_pc->positions()[k]).eval();
    p_mean_tmp = (p_mean_tmp) / tmp_pc->num_points();    
    for (int k = 0; k < covis_pc->num_points(); k++)
      p_mean_covis = (p_mean_covis + covis_pc->positions()[k]).eval();
    p_mean_covis = (p_mean_covis) / covis_pc->num_points();
    float dist = (p_mean_covis - p_mean_tmp).norm();
    std::cout<<"dist between tmp and covis is "<<dist<<", p_mean_tmp is "<<p_mean_tmp
             <<", p_mean_covis is "<<p_mean_covis<<"\n";
    
    cvo::CvoParams & init_param = cvo_align.get_params();
    float ell_init = init_param.ell_init;
    float ell_decay_rate = init_param.ell_decay_rate;
    int ell_decay_start = init_param.ell_decay_start;
    init_param.ell_init = dist;  //init_param.ell_init_first_frame;
    init_param.ell_decay_rate = init_param.ell_decay_rate_first_frame;
    init_param.ell_decay_start  = init_param.ell_decay_start_first_frame;
    cvo_align.write_params(&init_param);

    Eigen::Matrix4f T_t2s = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f output;
    std::cout<<"start align....\n"<<std::flush;
    cvo_align.align(*tmp_pc, *covis_pc, T_t2s, output );
    std::cout<<"end align....\n";
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> init_pose = Eigen::Matrix<double, 4, 4, Eigen::RowMajor> ::Identity();
    init_pose.block<3,4>(0,0) = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> (frames[0]->pose_vec);
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_row_f =  output;
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_row = init_pose *  T_row_f.cast<double>();
    // Eigen::Map<Eigen::Matrix<double, 3,4, Eigen::RowMajor>>(covisFrames[1]->pose_vec ) = T_row.data();
    for (int k = 0; k < 12; k++)
      covisFrames[1]->pose_vec[k]  = T_row.data()[k];
    f_name = "after_covis_BA.pcd";
    write_transformed_pc(covisFrames, f_name);

    
    //frames.push_back(new_frame);
    //frames_full.push_back(new_frame);
    
    
    // std::cout<<"read number points is "<<pc->num_points()<<std::endl;

  }

  
  

  //std::string traj_out("traj_out.txt");
  //write_traj_file(traj_out,
  //               timestamps,
  //                frames );
 

  return 0;
}
