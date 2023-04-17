#include "cvo/CvoFrame.hpp"
#include "utils/CvoPointCloud.hpp"
#include <cstdlib>

namespace cvo {

  CvoFrame::CvoFrame(const CvoPointCloud * pts,
                     const double poses[12]) : points(pts)

                                               // points_transformed_(pts->num_features(), pts->num_classes()){
  {
    memcpy(pose_vec, poses, 12*sizeof(double));
  
  }

  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & CvoFrame::points_transformed() {
    return points_transformed_;
  }

  void CvoFrame::transform_pointcloud() {
    
  }

  std::shared_ptr<CvoPointCloudGPU> CvoFrame::points_init_gpu() const {
    
  }
    std::shared_ptr<CvoPointCloudGPU> points_transformed_gpu() const;
    const CuKdTree & kdtree() const;
   

}
