#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace cvo {

  class RawImage {
  public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    RawImage(const cv::Mat & left_image, bool is_denoise=true);
    RawImage(const cv::Mat & left_image, int num_classes, const std::vector<float> & semantic, bool is_denoise=true);
    RawImage();
    virtual ~RawImage() {}

    const std::vector<float> & intensity() const { return intensity_; }
    cv::Mat intensity_to_img() const;
    const cv::Mat & image() const { return image_;}
    const std::vector<float> & gradient() const {return gradient_;}
    const std::vector<float> & gradient_square() const {return gradient_square_;}
    const std::vector<float> & semantic_image() const {return semantic_image_;}
    cv::Mat semantic_to_img() const;
    int rows() const {return rows_;}
    int cols() const {return cols_;}
    int channels() const {return channels_;}
    int num_classes() const {return num_class_;}

    void add_color_noise(float noise_ratio, float noise_sigma);
    void add_semantic_noise(float noise_ratio, float noise_sigma);

    //virtual void add_blur_noise(float sigma);
    
  private: 
    // assume all data to be float32
    cv::Mat image_;
    std::vector<float> intensity_;
    std::vector<float> gradient_; // size: image total pixels x 2
    std::vector<float> gradient_square_;
    int num_class_;
    int rows_; 
    int cols_;
    int channels_;
    std::vector<float> semantic_image_;

    // fill in gradient_ and gradient_square_
    void compute_image_gradient();
    
    
  };

}
