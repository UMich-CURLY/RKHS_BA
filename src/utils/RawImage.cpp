#include <cmath>
#include <cstdint>
#include <random>
#include <opencv2/photo.hpp>
#include "utils/RawImage.hpp"
#include "utils/GaussianMixture.hpp"

namespace cvo {
  RawImage::RawImage(const cv::Mat & image, bool is_denoise)
    : gradient_(image.total() * 2, 0),
      gradient_square_(image.total(), 0) {

    if (image.total() == 0)
      return;
    
    num_class_ = 0;
    // std::cout<<"Start clone: image "<<image.rows<<"x"<<image.cols<<", channel is "<<image.channels()<<"\n";
    image_ = image.clone();
    rows_ = image.rows;
    cols_ = image.cols;
    channels_ = image.channels();
    //std::cout<<"Raw Image: channels is "<< image.channels()<<"\n";
    if (is_denoise) {
      if (channels_==3)
        cv::fastNlMeansDenoisingColored(image_,image_,10,10,7,21);
      else if (channels_ == 1)
        cv::fastNlMeansDenoising(image_,image_,10,7,21);
      else {
        std::cerr<<"Image channels should be 1 or 3!\n";
        return;
      }
    }
    //std::cout<<"denoising complete\n";
    //cv::fastNlMeansDenoising (color_, color_);
    intensity_.resize(image_.total());
    cv::Mat gray;    
    if (channels_ == 3) {
      cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
      gray.convertTo(gray, CV_32FC1);
    } else {
      image.convertTo(gray, CV_32FC1);
    }
    memcpy(intensity_.data(), gray.data, sizeof(float) * image_.total());      
    compute_image_gradient();
    //std::cout<<"Raw Image created\n";
  }

  RawImage::RawImage(const cv::Mat & image, 
                     int num_classes, const std::vector<float> & semantic,
                     bool is_denoise)
    : RawImage(image)  {
    num_class_ = num_classes;
    semantic_image_.resize(semantic.size());
    memcpy(semantic_image_.data(), semantic.data(), sizeof(float) * semantic.size() );
  }

  RawImage::RawImage(){
    
  }

  void RawImage::add_color_noise(float noise_ratio, float noise_sigma) {
    if (noise_sigma > 1e-4 && noise_ratio > 1e-4) {
      cvo::GaussianMixtureDepthGenerator noise(noise_ratio, noise_sigma, 0);
      #pragma omp parallel for
      for (int c = 0; c < cols_; c++) {
        for (int r = 0; r < rows_; r++) {
          //uint8_t noise[3];
          cv::Vec3b & pixel = image_.at<cv::Vec3b>(r, c);
          for (int j = 0; j < channels_; j++)  {
            float  pix = (float)pixel[j];
            pix = noise.sample<float>(pix, 1.0);
            if (pix < 0)
              pix = 0.0;
            if (pix > 255.0)
              pix = 255.0;
            pixel[j] = (uint8_t)pix;
          }
          //std::cout<<"After perturbation, semantic_dist is "<<semantic_dist.transpose()<<"\n";
        }
      }
      compute_image_gradient();
    }
    
  }
  void RawImage::add_semantic_noise(float noise_ratio, float noise_sigma) {
    if (semantic_image_.size() && noise_sigma > 1e-4 && noise_sigma > 1e-4) {
        
      cvo::GaussianMixtureDepthGenerator noise(noise_ratio, noise_sigma, 0);
        
      //#pragma omp parallel for
      for (int c = 0; c < cols_; c++) {
        for (int r = 0; r < rows_; r++) {

          for (int j = 0; j < num_class_; j++)  {
            float & semantic = semantic_image_[(r*cols_+c)*num_class_+j];            
            semantic = noise.sample<float>(semantic, 1.0); ///(semantic_dist[j] + dist_semantic(generator));
            if (semantic < 0)
              semantic = 0;
          }
          Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> semantic_dist(&semantic_image_[(r*cols_+c)*num_class_],
                                                                            num_class_);           
          semantic_dist.normalize();
          //std::cout<<"After perturbation, semantic_dist is "<<semantic_dist.transpose()<<"\n";
          if (c == 1 && r == 1) {
            std::cout<<"normalized semantic dist with noise is: ";
            std::cout<<semantic_dist.transpose();
            std::cout<<"\n";
          }
        }
      }

    }   
    
  }
  

  void RawImage::compute_image_gradient() {


    // calculate gradient
    // we skip the first row&col and the last row&col

    for(int idx=cols_; idx<cols_*(rows_-1); idx++) {
      if (idx % cols_ == 0 || idx%cols_ == cols_-1) {
        gradient_[idx * 2] = 0;
        gradient_[idx * 2 + 1] = 0;
        gradient_square_[idx] = 0;
        continue;
      }
                
      float dx = 0.5f*( intensity_[idx+1] - intensity_[idx-1] );
      float dy = 0.5f*( intensity_[idx+cols_] - intensity_[idx-cols_] );

      // if it's not finite, set to 0
      if(!std::isfinite(dx)) dx=0;
      if(!std::isfinite(dy)) dy=0;
                
      gradient_[2*idx] = dx;
      gradient_[2*idx+1] = dy;
      gradient_square_[idx] = dx*dx+dy*dy;

    }
    
  }

  cv::Mat RawImage::intensity_to_img() const {
    //std::vector<float> intensity(intensity_.size());
    //std::memcpy(intensity.data(), intensity_.data(), intensity_.size()*sizeof(float));
    //cv::Mat image_float(cv::Size(cols_, rows_), CV_32FC3, intensity.data());
    /*
    cv::Mat image(cv::Size(cols_, rows_), CV_8UC3);
    for (int c = 0; c < cols_; c++) {
      for (int r = 0; r < rows_; r++) {
        if (channels_ == 3) {
          cv::Vec3b color;
          for (int i = 0; i < channels_; i++)
            color[i] = (uint8_t)intensity_[(r*cols_+c)*channels_+i];
          image.at<cv::Vec3b>(r, c) = color;
        }
        else if (channels_ == 1) {
          image.at<uint8_t>(r, c) = (uint8_t)intensity_[(r*cols_+c)*channels_];
        }
        
      }
      }*/
    //image_float.convertTo(image, CV_8UC3);
    //cv::imwrite("ing_noise.jpg", image);

    return image_.clone();
  }

  cv::Mat RawImage::semantic_to_img() const {
    std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
    label2color[0]  =std::make_tuple(128, 64,128 ); // road
    label2color[1]  =std::make_tuple(244, 35,232 ); // sidewalk
    label2color[2]  =std::make_tuple(70, 70, 70 ); // sidewalk
    label2color[3]  =std::make_tuple(102,102,156   ); // building
    label2color[4] =std::make_tuple(190,153,153 ); // pole
    label2color[5] =std::make_tuple(153,153,153  ); // sign
    label2color[6]  =std::make_tuple(250,170, 30   ); // vegetation
    label2color[7]  =std::make_tuple(220,220,  0   ); // terrain
    label2color[8] =std::make_tuple(107,142, 35 ); // sky
    label2color[9]  =std::make_tuple(152,251,152 ); // water
    label2color[10]  =std::make_tuple(70,130,180  ); // person
    label2color[11]  =std::make_tuple( 220, 20, 60   ); // car
    label2color[12]  =std::make_tuple(255,  0,  0  ); // bike
    label2color[13] =std::make_tuple( 0,  0,142 ); // stair
    label2color[14]  =std::make_tuple(0,  0, 70 ); // background
    label2color[15]  =std::make_tuple(0, 60,100 ); // background
    label2color[16]  =std::make_tuple(0, 80,100 ); // background
    label2color[17]  =std::make_tuple( 0,  0,230 ); // background
    label2color[18]  =std::make_tuple(119, 11, 32 ); // background

    cv::Mat label_in_color(rows_, cols_, CV_8UC3);
    for (int c = 0; c < cols_; c++) {
      for (int r = 0; r < rows_; r++) {
        Eigen::Map<const Eigen::VectorXf> dist(&(this->semantic_image_[(r*cols_+c)*num_class_]), num_class_);
        int label = dist.maxCoeff();
        auto color = label2color[label];
        cv::Vec3b pix;
        pix[0] = std::get<0>(color);
        pix[1] = std::get<1>(color);
        pix[2] = std::get<2>(color);
        label_in_color.at<cv::Vec3b>(r, c) = pix;
      }
    }
    
    
    return label_in_color;    
  }

}
