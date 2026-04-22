#pragma once
#include "DataHandler.hpp"
#include <string>
#include <filesystem>
#include <algorithm>
#include <pcl/io/pcd_io.h>

namespace cvo {
  class PcdHandler : public DatasetHandler{
  public:
    PcdHandler(){}
    PcdHandler(const std::string & folder ) : folder_name(folder) {
      for(auto & p : std::filesystem::directory_iterator( folder ) ) {
        // If it's not a directory, list it. If you want to list directories too, just remove this check.
        if (std::filesystem::is_regular_file(p.path())) {
          // assign current file name to current_file and echo it out to the console.
          std::string current_file = p.path().string();
          const std::string pcd = ".pcd";
          if (current_file.size() >= pcd.size() && current_file.compare(current_file.size()-pcd.size(), pcd.size(), pcd) == 0) {
            files.push_back(current_file);
          }
          // cout <<"reading "<< current_file << endl; 
        }
      }

      std::sort(files.begin(), files.end());

      curr_index  = 0;

    }

    void set_start_index(int start) { curr_index = start; }
    int get_current_index() { return curr_index; }
    int get_total_number() { return files.size(); }
    void next_frame_index() { 
	    if (curr_index < files.size() - 1) 
	        curr_index++; 
    }

    int read_next_pcd(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
	   if (pc == nullptr ||
			   curr_index >= files.size())
		  return -1; 
   	pcl::io::loadPCDFile(files[curr_index], *pc);
	return 0;
    }
    

  private:
    int curr_index;
    std::string folder_name;
    std::vector<std::string> files;
    
    
  };
}
