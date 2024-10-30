#pragma once

#include <vector>
#include "RawImage.hpp"

namespace cv {
  class Mat;
}

namespace cvo {
  template <typename DepthType>
  class ImageRGBD : public RawImage {
  public:
    ImageRGBD(const cv::Mat & image,
              const std::vector<DepthType> & depth_image,
              bool is_denoise=true) : depth_image_(depth_image),
                                      RawImage(image, is_denoise) {}

    ImageRGBD(const cv::Mat & image,
              const std::vector<DepthType> & depth_image,
              int num_classes,
              const std::vector<float> & semantics,
              bool is_denoise=true) : depth_image_(depth_image),
                                      RawImage(image, num_classes, semantics, is_denoise) {}

    const std::vector<DepthType> & depth_image() const  { return depth_image_; }
    
  private:
    std::vector<DepthType>  depth_image_;
  };
  
}
