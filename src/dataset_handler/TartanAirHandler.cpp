#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include "dataset_handler/TartanAirHandler.hpp"

using namespace std;
namespace fs = std::filesystem;

namespace cvo {
  namespace {

  template <typename T>
  std::vector<T> load_npy_flat(const std::string& path, std::vector<size_t>& shape) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
      throw std::runtime_error("Failed to open npy file: " + path);
    }

    char magic[6];
    in.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
      throw std::runtime_error("Invalid npy magic: " + path);
    }

    char version[2];
    in.read(version, 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
      uint16_t h16 = 0;
      in.read(reinterpret_cast<char*>(&h16), sizeof(h16));
      header_len = h16;
    } else {
      in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }

    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    const auto shape_pos = header.find("shape");
    const auto open = header.find('(', shape_pos);
    const auto close = header.find(')', open);
    if (shape_pos == std::string::npos || open == std::string::npos || close == std::string::npos) {
      throw std::runtime_error("Failed to parse npy shape: " + path);
    }

    std::stringstream ss(header.substr(open + 1, close - open - 1));
    shape.clear();
    while (ss.good()) {
      std::string token;
      std::getline(ss, token, ',');
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
      if (!token.empty()) {
        shape.push_back(static_cast<size_t>(std::stoul(token)));
      }
    }
    if (shape.empty()) {
      throw std::runtime_error("Empty npy shape: " + path);
    }

    size_t total = 1;
    for (size_t dim : shape) {
      total *= dim;
    }
    std::vector<T> data(total);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(T)));
    if (!in.good()) {
      throw std::runtime_error("Failed to read npy payload: " + path);
    }
    return data;
  }

  } // namespace

  TartanAirHandler::TartanAirHandler(std::string tartan_traj_folder,
                                     std::string depth_folder_dir){
    this->folder_name = tartan_traj_folder;

    if (depth_folder_dir.size() == 0)
      depth_folder_dir = "depth_left";
      
    this->depth_folder_name = depth_folder_dir;
    // use left camera only, rgbd
    const string depth_pth = tartan_traj_folder + "/" + this->depth_folder_name;
    const string image_pth = tartan_traj_folder + "/image_left";
    // count number of files in both dirs
    int depth_count = 0;
    fs::directory_iterator end_it;
    for (fs::directory_iterator it(depth_pth); it != end_it; it++) {
      depth_count++;
    }
    int image_count = 0;
    for (fs::directory_iterator it(image_pth); it != end_it; it++) {
      image_count++;
    }
    //assert (depth_count == image_count);
    total_size = depth_count;
    curr_index = 0;

    cout << "Found " << total_size << " image pairs\n";
    cout << "Searching for semantic class mapping file\n";
    std::ifstream map_file(tartan_traj_folder + "/seg_map.txt");
    if (!map_file.good()) {
      cout << "Couldn't parse mapping file\n";
    } else {
      while (map_file) {
        string line;
        if (!getline(map_file, line)) break;
        int colon_idx = line.find(':');
        uint8_t key = uint8_t(stoi(line.substr(0, colon_idx)));
        uint8_t val = uint8_t(stoi(line.substr(colon_idx + 1)));
        // cout << key << " : " << val << endl;
        semantic_class[key] = val;
      }
      cout << "Parsed mapping file!\n";
    }
  }

  TartanAirHandler::~TartanAirHandler() {}
  
  int TartanAirHandler::read_next_rgbd(cv::Mat & rgb_img, cv::Mat & dep_img) {
    if (curr_index >= total_size)
      return -1;
    // format curr_index
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();
    // read rgb image
    string img_pth = folder_name + "/image_left/" + index_str + "_left.png";
    rgb_img = cv::imread(img_pth, cv::ImreadModes::IMREAD_COLOR);
    // read depth npy
    string dep_pth = folder_name + "/" + depth_folder_name + "/" + index_str + "_left_depth.png";
    std::cout<<"Read depth"<< dep_pth<<"\n";
    dep_img = cv::imread(dep_pth, cv::IMREAD_UNCHANGED);
    if (rgb_img.data == nullptr || dep_img.data == nullptr) {
      cerr<<"Image doesn't read successfully: "<<img_pth<<", "<<dep_pth<<"\n";
      return -1;
    }
    
    /*
    cnpy::NpyArray dep_arr = cnpy::npy_load(dep_pth);
    float* dep_data = dep_arr.data<float>();
    int dim1 = dep_arr.shape[0];
    int dim2 = dep_arr.shape[1];
    cv::Mat raw_dep(cv::Size(dim2, dim1), CV_32FC1, dep_data);

    // set high depth pixels (sky) to nan
    #pragma omp parallel for
    for (int r = 0; r < raw_dep.rows; r++) {
      for (int c = 0; c < raw_dep.cols; c++) {
        if (raw_dep.at<float>(r, c) < 100) continue;
        raw_dep.at<float>(r, c) = std::nanf("1");
      }
    }
    */
    // scale by 5000 and convert to uint16_t
    //aw_dep = raw_dep * 5000.0f;
      //raw_dep.convertTo(dep_img, CV_16UC1);
    return 0;
  }

  int TartanAirHandler::read_next_rgbd(cv::Mat & rgb_img, std::vector<float> & dep_vec,
                                       float max_depth,
                                       bool is_disparity) {
    if (curr_index >= total_size) {
      return -1;
    }
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();

    string img_pth = folder_name + "/image_left/" + index_str + "_left.png";
    rgb_img = cv::imread(img_pth, cv::ImreadModes::IMREAD_COLOR);
    const string depth_folder = folder_name + "/" + depth_folder_name;
    const string npy_path = depth_folder + "/" + index_str + "_left_depth.npy";
    if (rgb_img.data == nullptr) {
      cerr<<"Image doesn't read successfully: "<<img_pth<<"\n";
      return -1;
    }

    std::vector<size_t> shape;
    const auto depth_data = load_npy_flat<float>(npy_path, shape);
    if (shape.size() != 2) {
      throw std::runtime_error("Expected 2D tartan depth npy: " + npy_path);
    }
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    dep_vec.resize(depth_data.size());
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        float pix = depth_data[r * cols + c];
        if (!std::isfinite(pix) || pix <= 0.0f || pix > 60000.0f) {
          pix = std::nanf("1");
        } else {
          if (is_disparity) {
            pix = 80.0f / pix;
          }
          if (pix > max_depth) {
            pix = std::nanf("1");
          }
        }
        dep_vec[r * cols + c] = pix;
      }
    }
    return 0;
  }

  int TartanAirHandler::read_next_rgbd(cv::Mat & rgb_img, std::vector<float> & dep_vec,
                                       int num_semantic_class, std::vector<float> & semantics,
                                       float max_depth,
                                       bool is_disparity) {
    if (this->sky_label_!=-1) {
      if (-1 == read_next_rgbd_without_sky(rgb_img,
                                           dep_vec,
                                           num_semantic_class,
                                           semantics,
                                           this->sky_label_,
                                           max_depth,
                                           is_disparity))

        return -1;
    } else {
      if (read_next_rgbd(rgb_img, dep_vec, max_depth, is_disparity))
        return -1;

      if (read_next_semantics(rgb_img.total(), num_semantic_class, semantics))
        return -1;

    }
    return 0;    
  }

  int TartanAirHandler::read_next_stereo(cv::Mat & left, cv::Mat & right) {
    if (curr_index >= total_size)
      return -1;
    // format curr_index
    stringstream ss;
    ss << setw(6) << setfill('0') << curr_index;
    string index_str = ss.str();
    // read l, r images
    string left_pth = folder_name + "/image_left/" + index_str + "_left.png";
    string right_pth = folder_name + "/image_right/" + index_str + "_right.png";
    left = cv::imread(left_pth, cv::ImreadModes::IMREAD_COLOR);
    right = cv::imread(right_pth, cv::ImreadModes::IMREAD_COLOR);
    if (left.data == nullptr || right.data == nullptr) {
      cerr << "Image doesn't read successfully: " << left_pth << ", " << right_pth << "\n";
      return -1;
    }
    return 0;
  }

  int TartanAirHandler::read_next_stereo(cv::Mat & left, cv::Mat & right,
                                         int num_semantic_class,
                                         std::vector<float> & semantics) {
    if (read_next_stereo(left, right))
      return -1;
    if (read_next_semantics(left.total(), num_semantic_class, semantics))
      return -1;
    return 0;
  }

  void TartanAirHandler::next_frame_index() {
    curr_index++;
  }

  void TartanAirHandler::set_depth_folder_name(const std::string & folder) {
    this->depth_folder_name = folder;
  }


  void TartanAirHandler::set_start_index(int start) {
    curr_index = start;
  }

  int TartanAirHandler::get_current_index() {
    return curr_index;
  }

  int TartanAirHandler::get_total_number() {
    return total_size;
  }

  int TartanAirHandler::read_next_semantics(int num_pixels, int num_semantic_class, std::vector<float> & semantics) {
    (void)num_pixels;
    (void)num_semantic_class;
    semantics.clear();
    return -1;
  }


  int TartanAirHandler::read_next_rgbd_without_sky(cv::Mat & rgb_img,
                                                   std::vector<float> & dep_vec,
                                                   int num_semantic_class,
                                                   std::vector<float> & semantics,
                                                   int sky_label,
                                                   float max_depth,
                                                   bool is_disparity) {
    
    if (read_next_rgbd(rgb_img, dep_vec, max_depth, is_disparity))
      return -1;
    
    (void)num_semantic_class;
    (void)sky_label;
    (void)max_depth;
    semantics.clear();
    return -1;
    
  }
  

}
