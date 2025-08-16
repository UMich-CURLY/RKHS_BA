#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <filesystem>
#include "utils/CvoPointCloud.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/Cvo.hpp"
#include "cvo/CvoParams.hpp"
#include "utils/CvoPoint.hpp"
#include "utils/PointSegmentedDistribution.hpp"
#include "dataset_handler/DataHandler.hpp"
#include "dataset_handler/KittiHandler.hpp"
#include "dataset_handler/TartanAirHandler.hpp"
#include "utils/ImageRGBD.hpp"
#include <opencv2/opencv.hpp>
#include "utils/Calibration.hpp"

enum DatasetType {
  STEREO=0,  
  LIDAR,
  RGBD
};

int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  std::string dataset_name(argv[1]);
  std::string dataset_path(argv[2]);
  DatasetType dataset_type = static_cast<DatasetType>(std::stoi(argv[3])); // LIDAR, RGBD, STEREO
  std::string out_prefix(argv[4]);
  int start_frame = std::stoi(argv[5]);
  int every_n_frame = std::stoi(argv[6]);
  int num_frames = std::stoi(argv[7]);
  std::string calib_file;
  std::unique_ptr<cvo::Calibration> calib;
  if (argc > 8) {
    calib_file = (argv[8]);
    if (dataset_type == STEREO)
      calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::PointCloudType::STEREO));
    else
      calib.reset(new cvo::Calibration(calib_file, cvo::Calibration::PointCloudType::RGBD));      
  }

  std::unique_ptr<cvo::DatasetHandler> dataset;  
  if (std::strcmp(dataset_name.c_str(), "kitti") == 0) {
    dataset.reset(new cvo::KittiHandler(dataset_path, static_cast<cvo::KittiHandler::DataType>(dataset_type)));
  } else if (std::strcmp(dataset_name.c_str(), "tartanair") == 0) {
    dataset.reset(new cvo::TartanAirHandler(dataset_path));
  } else {
    ASSERT(false, "unknown_dataset "+dataset_name);
  }


  for (int i = 0; i < num_frames; i++) {
    dataset->set_start_index(start_frame + i*every_n_frame);
    if (dataset_type == STEREO) {
      
    } else if (dataset_type == RGBD) {
      cv::Mat left;
      std::vector<float> depth;
      dataset->read_next_rgbd(left, depth);
      cvo::ImageRGBD<float> image(left, depth, false);
      cvo::CvoPointCloud cvo_pc(image, *calib);
      pcl::PointCloud<pcl::PointXYZRGB> pc;
      cvo_pc.export_to_pcd<pcl::PointXYZRGB>(pc);
      pcl::io::savePCDFileASCII(out_prefix+std::to_string(start_frame+i*every_n_frame)+".pcd", pc);
      
    } else if (dataset_type == LIDAR) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
      dataset->read_next_lidar(cloud);
      pcl::io::savePCDFileASCII(out_prefix+std::to_string(start_frame+i*every_n_frame)+".pcd", *cloud);
    } else {
      std::cerr<<" Unknown dataset type \n";
      exit(-1);
    }
  }

  return 0;
}
