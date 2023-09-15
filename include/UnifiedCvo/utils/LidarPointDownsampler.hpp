#pragma once
#include <iostream>
#include "utils/LidarPointSelector.hpp"
#include "utils/LidarPointType.hpp"
#include "utils/VoxelMap.hpp"
#include "utils/data_type.hpp"
#include <unordered_set>

namespace cvo {

  
  std::shared_ptr<cvo::CvoPointCloud> downsample_lidar_points(bool is_edge_only,
                                                              pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in,
                                                              float leaf_size) {


    int expected_points = 5000;
    double intensity_bound = 0.4;
    double depth_bound = 4.0;
    double distance_bound = 40.0;
    int kitti_beam_num = 64;
    cvo::LidarPointSelector lps(expected_points, intensity_bound, depth_bound, distance_bound, kitti_beam_num);
    /*
    // running edge detection + lego loam point selection
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_surface (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<int> selected_edge_inds, selected_loam_inds;
    lps.edge_detection(pc, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
  
    lps.legoloam_point_selector(pc, pc_out_surface, edge_or_surface, selected_loam_inds);    
    // *pc_out += *pc_out_edge;
    // *pc_out += *pc_out_surface;
    //
    num_points_ = selected_indexes.size();
    */

    if (is_edge_only) {
      cvo::VoxelMap<pcl::PointXYZI> full_voxel(leaf_size);
      for (int k = 0; k < pc_in->size(); k++) {
        full_voxel.insert_point(&pc_in->points[k]);
      }
      std::vector<pcl::PointXYZI*> downsampled_results = full_voxel.sample_points();
      pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
      for (int k = 0; k < downsampled_results.size(); k++)
        downsampled->push_back(*downsampled_results[k]);
      std::shared_ptr<cvo::CvoPointCloud>  ret(new cvo::CvoPointCloud(downsampled, 5000, 64, cvo::CvoPointCloud::PointSelectionMethod::FULL));
      return ret;
    } else {
    
      /// edge points
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_edge (new pcl::PointCloud<pcl::PointXYZI>);  
      std::vector<int> selected_edge_inds;
      std::vector <double> output_depth_grad;
      std::vector <double> output_intenstity_grad;
      lps.edge_detection(pc_in, pc_out_edge, output_depth_grad, output_intenstity_grad, selected_edge_inds);
      std::unordered_set<int> edge_inds;
      for (auto && j : selected_edge_inds) edge_inds.insert(j);

      /// surface points
      std::vector<float> edge_or_surface;
      std::vector<int> selected_loam_inds;
      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out_loam (new pcl::PointCloud<pcl::PointXYZI>);        
      lps.loam_point_selector(pc_in, pc_out_loam, edge_or_surface, selected_loam_inds);

      /// declare voxel map
      cvo::VoxelMap<pcl::PointXYZI> edge_voxel(leaf_size); 
      cvo::VoxelMap<pcl::PointXYZI> surface_voxel(leaf_size / 2);

      /// edge and surface downsample
      for (int k = 0; k < pc_out_edge->size(); k++) 
        edge_voxel.insert_point(&pc_out_edge->points[k]);
      std::vector<pcl::PointXYZI*> edge_results = edge_voxel.sample_points();
      //for (int k = 0; k < pc_in->size(); k++)  {
      //  if (edge_or_surface[k] > 0 &&
      //      edge_inds.find(k) == edge_inds.end())
      //    surface_voxel.insert_point(&pc_in->points[k]);
      // }
      for (int k = 0; k < pc_out_loam->size(); k++)  {
        if (edge_or_surface[k] > 0 &&
            edge_inds.find(selected_loam_inds[k]) == edge_inds.end())
          surface_voxel.insert_point(&pc_out_loam->points[k]);
      }
      std::vector<pcl::PointXYZI*> surface_results = surface_voxel.sample_points();
      int total_selected_pts_num = edge_results.size() + surface_results.size();    
      std::shared_ptr<cvo::CvoPointCloud> ret(new cvo::CvoPointCloud(1, NUM_CLASSES));
      ret->reserve(total_selected_pts_num, 1, NUM_CLASSES);
      std::cout<<"edge voxel selected points "<<edge_results.size()<<std::endl;
      std::cout<<"surface voxel selected points "<<surface_results.size()<<std::endl;    

      /// push
      for (int k = 0; k < edge_results.size(); k++) {
        Eigen::VectorXf feat(1);
        feat(0) = edge_results[k]->intensity;
        Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
        Eigen::VectorXf geo_t(2);
        geo_t << 1.0, 0;
        ret->add_point(k, edge_results[k]->getVector3fMap(),  feat, semantics, geo_t);
      }
      /// surface downsample
      for (int k = 0; k < surface_results.size(); k++) {
        // surface_pcl.push_back(*surface_results[k]);
        Eigen::VectorXf feat(1);
        feat(0) = surface_results[k]->intensity;
        Eigen::VectorXf semantics = Eigen::VectorXf::Zero(NUM_CLASSES);
        Eigen::VectorXf geo_t(2);
        geo_t << 0, 1.0;
        ret->add_point(k+edge_results.size(), surface_results[k]->getVector3fMap(), feat,
                       semantics, geo_t);
      }
      return ret;

    }

  }


}
