#include "utils/CvoPointCloud.hpp"
#include "utils/PointSegmentedDistribution.hpp"
//#include "utils/CvoFrameGPU.hpp"
#include "utils/CvoPoint.hpp"
#include "cvo/nanoflann.hpp"
#include "cvo/CvoParams.hpp"
#include "cvo/CvoGPU.hpp"
#include "cvo/CvoFrame.hpp"
//#include "cvo/CvoFrameGPU.hpp"
#include "cvo/IRLS_State.hpp"
#include "cvo/IRLS_State_CPU.hpp"
#include "cvo/IRLS_State_GPU.hpp"
#include "cvo/KDTreeVectorOfVectorsAdaptor.h"
#include "cvo/IRLS.hpp"
#include <tbb/tbb.h>
#include <memory>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <chrono>
#include <sophus/se3.hpp>

namespace cvo {
  typedef Eigen::Triplet<float> Trip_t;


  // template <Eigen::StorageOptions option>
  double dist_se3_cpu(const Eigen::Matrix<double, 4,4, Eigen::DontAlign> & m ) {
    Eigen::Matrix4d m_eig = m;
    Sophus::SE3d dRT_sophus(m_eig);
    double dist_this_iter = dRT_sophus.log().norm();
    return dist_this_iter;
  }

  //template
  //double dist_se3_cpu<Eigen::DontAlign>(const Eigen::Matrix<double,4,4, Eigen::DontAlign> & m);

  
  void CvoPointCloud_to_pcl(const CvoPointCloud & cvo_cloud,
                            pcl::PointCloud<CvoPoint> &pcl_cloud
                            ) {
    int num_points = cvo_cloud.num_points();
//    const ArrayVec3f & positions = cvo_cloud.positions();
//    const Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIMENSIONS> & features = cvo_cloud.features();
    //const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> & normals = cvo_cloud.normals();
    // const Eigen::Matrix<float, Eigen::Dynamic, 2> & types = cvo_cloud.types();
//    auto & labels = cvo_cloud.labels();
    // set basic informations for pcl_cloud
    pcl_cloud.resize(num_points);

    //int actual_num = 0;
    for(int i=0; i<num_points; ++i){
      //memcpy(&host_cloud[i], &cvo_cloud[i], sizeof(CvoPoint));
      //TODO: Might be problematic, refering to the change in CvoPointCloud_to_GPU in CvoGPU_impl.cu
      CvoPoint point = cvo_cloud.point_at(i);
      (pcl_cloud)[i].x = point.x;
      (pcl_cloud)[i].y = point.y;
      (pcl_cloud)[i].z = point.z;
//      if (FEATURE_DIMENSIONS >= 3 && features.cols() > 2 && features.rows() > i-1) {
      if (FEATURE_DIMENSIONS >= 3) {
          (pcl_cloud)[i].r = (uint8_t)std::min(255.0, (point.features[0] * 255.0));
          (pcl_cloud)[i].g = (uint8_t)std::min(255.0, (point.features[1] * 255.0));
          (pcl_cloud)[i].b = (uint8_t)std::min(255.0, (point.features[2] * 255.0));
      }

      for (int j = 0; j < FEATURE_DIMENSIONS; j++)
        pcl_cloud[i].features[j] = point.features[j];

      if (cvo_cloud.num_classes() > 0) {
//        labels.row(i).maxCoeff(&pcl_cloud[i].label);
        cvo_cloud.label_at(i).maxCoeff(&pcl_cloud[i].label);
        for (int j = 0; j < cvo_cloud.num_classes(); j++)
          pcl_cloud[i].label_distribution[j] = point.label_distribution[j];
      }

      pcl_cloud[i].geometric_type[0] = point.geometric_type[0];
      pcl_cloud[i].geometric_type[1] = point.geometric_type[1];
      
      //if (normals.rows() > 0 && normals.cols()>0) {
      //  for (int j = 0; j < 3; j++)
      //    pcl_cloud[i].normal[j] = normals(i,j);
      //}

      //if (cvo_cloud.covariance().size() > 0 )
      //  memcpy(pcl_cloud[i].covariance, cvo_cloud.covariance().data()+ i*9, sizeof(float)*9  );
      //if (cvo_cloud.eigenvalues().size() > 0 )
      //  memcpy(pcl_cloud[i].cov_eigenvalues, cvo_cloud.eigenvalues().data() + i*3, sizeof(float)*3);

      //if (i == 1000) {
      //  printf("Total %d, Raw input from pcl at 1000th: \n", num_points);
      //  print_point(pcl_cloud[i]);
      //}
      
    }
    //gpu_cloud->points = host_cloud;

    /*
      #ifdef IS_USING_COVARIANCE    
      auto covariance = &cvo_cloud.covariance();
      auto eigenvalues = &cvo_cloud.eigenvalues();
      thrust::device_vector<float> cov_gpu(cvo_cloud.covariance());
      thrust::device_vector<float> eig_gpu(cvo_cloud.eigenvalues());
      copy_covariances<<<host_cloud.size()/256 +1, 256>>>(thrust::raw_pointer_cast(cov_gpu.data()),
      thrust::raw_pointer_cast(eig_gpu.data()),
      host_cloud.size(),
      thrust::raw_pointer_cast(gpu_cloud->points.data()));
      #endif    
    */
    return;
  }

  
  static
  void se_kernel_init_ell_cpu(const CvoPointCloud* cloud_a, const CvoPointCloud* cloud_b, \
                              cloud_t* cloud_a_pos, cloud_t* cloud_b_pos, \
                              Eigen::SparseMatrix<float,Eigen::RowMajor>& A_temp,
                              tbb::concurrent_vector<Trip_t> & A_trip_concur_,
                              const CvoParams & params,
                              float ell
                              ) {
    bool debug_print = false;
    A_trip_concur_.clear();
    const float s2= params.sigma*params.sigma;

    const float l = ell;

    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    const float d2_thres = -2.0*l*l*log(params.sp_thres/s2);
    if (debug_print ) std::cout<<"l is "<<l<<",d2_thres is "<<d2_thres<<std::endl;
    const float d2_c_thres = -2.0*params.c_ell*params.c_ell*log(params.sp_thres/params.c_sigma/params.c_sigma);
    if (debug_print) std::cout<<"d2_c_thres is "<<d2_c_thres<<std::endl;
    
    typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;
    kd_tree_t mat_index(3 , (*cloud_b_pos), 50  );
    mat_index.index->buildIndex();
    // loop through points
    tbb::parallel_for(int(0),cloud_a->num_points(),[&](int i){
        //for(int i=0; i<num_fixed; ++i){
        const float search_radius = d2_thres;
        std::vector<std::pair<size_t,float>>  ret_matches;
        nanoflann::SearchParams params_flann;
        //params.sorted = false;
        const size_t nMatches = mat_index.index->radiusSearch(&(*cloud_a_pos)[i](0), search_radius, ret_matches, params_flann);
        Eigen::Matrix<float,Eigen::Dynamic,1> feature_a = cloud_a->feature_at(i).transpose();
        Eigen::VectorXf label_a;
        if (params.is_using_semantics)
          label_a = cloud_a->label_at(i);
        
        // for(int j=0; j<num_moving; j++){
        for(size_t j=0; j<nMatches; ++j){
          int idx = ret_matches[j].first;
          float d2 = ret_matches[j].second;
          // d2 = (x-y)^2
          float k = 1;
          float ck = 1;
          float sk = 1;
          float d2_color = 0;
          float d2_semantic = 0;
          float geo_sim=1;
          float a = 1;

          Eigen::VectorXf label_b;
          /*
//TODO: add geometric_types
          if (params.is_using_geometric_type) {
            geo_sim = ( p_a->geometric_type[0] * p_a->geometric_type[0] +
                        p_a->geometric_type[1] * p_a->geometric_type[1] ) /
              sqrt( square_norm(p_a->geometric_type, 2) *
                    square_norm(p_a->geometric_type, 2));
            if(geo_sim < 0.01)
              continue;        
              }*/
          
          
          if (params.is_using_semantics) {
            label_b = cloud_b->label_at(idx);
            d2_semantic = ((label_a-label_b).squaredNorm());            
            sk = params.s_sigma*params.s_sigma*exp(-d2_semantic/(2.0*params.s_ell*params.s_ell));
          }

          if (params.is_using_geometry) {
            k = s2*exp(-d2/(2.0*l*l));            
          } 

          if (params.is_using_intensity) {
            Eigen::Matrix<float,Eigen::Dynamic,1> feature_b = cloud_b->feature_at(idx).transpose();
            d2_color = ((feature_a-feature_b).squaredNorm());
            ck = params.c_sigma*params.c_sigma*exp(-d2_color/(2.0*params.c_ell*params.c_ell));
          }
          a = ck*k*sk*geo_sim;
          if (a > params.sp_thres){
            A_trip_concur_.push_back(Trip_t(i,idx,a));
          }
        }
      });

    A_temp.setFromTriplets(A_trip_concur_.begin(), A_trip_concur_.end());
    A_temp.makeCompressed();
  }



    
  float CvoGPU::inner_product_cpu(const CvoPointCloud& source_points,
                                  const CvoPointCloud& target_points,
                                  const Eigen::Matrix4f & t2s_frame_transform,
                                  float ell
                                  ) const {
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return 0;
    }
    ArrayVec3f fixed_positions = source_points.positions();
    ArrayVec3f moving_positions = target_points.positions();

    Eigen::Matrix4f s2t_frame_transform  = t2s_frame_transform.inverse();
    Eigen::Matrix3f rot = s2t_frame_transform.block<3,3>(0,0) ;
    Eigen::Vector3f trans = s2t_frame_transform.block<3,1>(0,3) ;
    // transform moving points
    tbb::parallel_for(int(0), target_points.num_points(), [&]( int j ){
      moving_positions[j] = (rot*moving_positions[j]+trans).eval();
    });

    Eigen::SparseMatrix<float,Eigen::RowMajor> A_mat;
    tbb::concurrent_vector<Trip_t> A_trip_concur_;
    A_trip_concur_.reserve(target_points.num_points() * 20);
    A_mat.resize(source_points.num_points(), target_points.num_points());
    A_mat.setZero();
    se_kernel_init_ell_cpu(&source_points, &target_points, &fixed_positions, &moving_positions, A_mat, A_trip_concur_ , params, ell );

    return A_mat.sum();
  }

  /*
  void CvoGPU::compute_association_gpu(const CvoPointCloud& source_points,
                                       const CvoPointCloud& target_points,
                                       const Eigen::Matrix4f & t2s_frame_transform,
                                       // output
                                       Association & association                                       
                                       ) const {
    
    if (source_points.num_points() == 0 || target_points.num_points() == 0) {
      return;
    }
    ArrayVec3f fixed_positions = source_points.positions();
    ArrayVec3f moving_positions = target_points.positions();

    Eigen::Matrix4f s2t_frame_transform  = t2s_frame_transform.inverse();
    Eigen::Matrix3f rot = s2t_frame_transform.block<3,3>(0,0) ;
    Eigen::Vector3f trans = s2t_frame_transform.block<3,1>(0,3) ;
    // transform moving points
    tbb::parallel_for(int(0), target_points.num_points(), [&]( int j ){
      moving_positions[j] = (rot*moving_positions[j]+trans).eval();
    });
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_mat;
    tbb::concurrent_vector<Trip_t> A_trip_concur_;
    A_trip_concur_.reserve(target_points.num_points() * 20);
    A_mat.resize(source_points.num_points(), target_points.num_points());
    A_mat.setZero();
    se_kernel_init_ell_cpu(&source_points, &target_points, &fixed_positions, &moving_positions, A_mat, A_trip_concur_ , params );
    A_mat_out = A_mat;
    return;
    
  }
  */

  static
  void align_multi_cpu_impl(std::vector<CvoFrame::Ptr> & frames,
                            const std::vector<bool> & frames_to_hold_const,                    
                            const std::list<BinaryState::Ptr> & binary_states,
                            const CvoParams & params
                            ) {
    
    CvoBatchIRLS batch_irls_problem(frames, frames_to_hold_const,
                                    binary_states, &params);

    batch_irls_problem.solve();
  }

  static std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> gen_icosahedron_init_rots() {
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> pose(60);
    pose[0]<<0.33333295583724976, -0.5773499011993408, 0.7453563809394836, -0.9341723918914795, -0.30901727080345154, 0.17841044068336487, 0.1273227334022522, -0.755761444568634, -0.6423500180244446;
    pose[1]<<0.33333295583724976, -0.5773499011993408, 0.7453563809394836, 0.35682153701782227, 0.8090172410011292, 0.46708619594573975, -0.8726784586906433, 0.11026399582624435, 0.4756830334663391;
    pose[2]<<0.33333295583724976, -0.5773499011993408, 0.7453563809394836, 0.5773507952690125, -0.5000000596046448, -0.645496666431427, 0.7453556656837463, 0.6454975008964539, 0.16666696965694427;
    pose[3]<<-0.33333367109298706, -0.3568219542503357, 0.8726779222488403, -0.5773502588272095, 0.8090170621871948, 0.11026380211114883, -0.7453559041023254, -0.46708616614341736, -0.47568386793136597;
    pose[4]<<-0.33333367109298706, -0.3568219542503357, 0.8726779222488403, 0.9341722726821899, -9.079904828013241e-08, 0.3568224310874939, -0.1273220181465149, 0.9341724514961243, 0.33333322405815125;
    pose[5]<<-0.33333367109298706, -0.3568219542503357, 0.8726779222488403, -0.35682207345962524, -0.80901700258255, -0.46708622574806213, 0.8726778030395508, -0.4670862853527069, 0.14235062897205353;
    pose[6]<<-0.33333367109298706, 0.35682183504104614, 0.8726779818534851, 0.5773502588272095, 0.8090170621871948, -0.11026366800069809, -0.7453559041023254, 0.4670862853527069, -0.4756837785243988;
    pose[7]<<-0.33333367109298706, 0.35682183504104614, 0.8726779818534851, 0.3568219542503357, -0.8090171813964844, 0.4670861065387726, 0.8726779818534851, 0.4670861065387726, 0.14235077798366547;
    pose[8]<<-0.33333367109298706, 0.35682183504104614, 0.8726779818534851, -0.9341722130775452, 1.5040369305552304e-07, -0.3568224310874939, -0.12732212245464325, -0.9341724514961243, 0.33333298563957214;
    pose[9]<<0.33333295583724976, 0.5773497819900513, 0.7453565001487732, 0.9341723918914795, -0.30901721119880676, -0.17841050028800964, 0.1273227334022522, 0.7557615637779236, -0.642349898815155;
    pose[10]<<0.33333295583724976, 0.5773497819900513, 0.7453565001487732, -0.577350914478302, -0.5000000596046448, 0.6454966068267822, 0.7453556060791016, -0.6454975605010986, 0.1666669398546219;
    pose[11]<<0.33333295583724976, 0.5773497819900513, 0.7453565001487732, -0.3568213880062103, 0.8090173602104187, -0.4670860767364502, -0.8726783990859985, -0.11026392132043839, 0.47568291425704956;
    pose[12]<<0.7453553080558777, -5.828192684020905e-08, 0.6666674613952637, 0.0, -1.0, -8.742277657347586e-08, 0.6666674613952637, 6.516103212561575e-08, -0.7453553080558777;
    pose[13]<<0.7453553080558777, -5.828192684020905e-08, 0.6666674613952637, -0.5773509740829468, 0.5, 0.645496666431427, -0.3333337903022766, -0.8660253882408142, 0.37267762422561646;
    pose[14]<<0.7453553080558777, -5.828192684020905e-08, 0.6666674613952637, 0.5773510336875916, 0.4999999701976776, -0.6454966068267822, -0.33333370089530945, 0.866025447845459, 0.37267765402793884;
    pose[15]<<0.33333346247673035, -0.9341723322868347, 0.12732192873954773, -0.5773497819900513, -0.3090169131755829, -0.7557618021965027, 0.7453562617301941, 0.17841142416000366, -0.6423498392105103;
    pose[16]<<0.33333346247673035, -0.9341723322868347, 0.12732192873954773, -0.35682255029678345, -3.2008651373871544e-07, 0.9341722130775452, -0.872677743434906, -0.3568222224712372, -0.33333390951156616;
    pose[17]<<0.33333346247673035, -0.9341723322868347, 0.12732192873954773, 0.9341723918914795, 0.30901724100112915, -0.17841053009033203, 0.1273215115070343, 0.1784108281135559, 0.9756838083267212;
    pose[18]<<-0.7453562617301941, -0.5773501992225647, 0.33333292603492737, -0.577349841594696, 0.30901703238487244, -0.7557616829872131, 0.3333335518836975, -0.7557613253593445, -0.5636608600616455;
    pose[19]<<-0.7453562617301941, -0.5773501992225647, 0.33333292603492737, -3.725290298461914e-07, 0.4999999403953552, 0.866025447845459, -0.6666664481163025, 0.6454973816871643, -0.37267830967903137;
    pose[20]<<-0.7453562617301941, -0.5773501992225647, 0.33333292603492737, 0.5773502588272095, -0.8090170621871948, -0.1102638840675354, 0.3333328664302826, 0.11026396602392197, 0.9363391995429993;
    pose[21]<<-0.7453562617301941, 0.5773501992225647, 0.33333301544189453, 3.725290298461914e-07, 0.5000000596046448, -0.8660253882408142, -0.6666664481163025, -0.6454972624778748, -0.3726784288883209;
    pose[22]<<-0.7453562617301941, 0.5773501992225647, 0.33333301544189453, 0.577349841594696, 0.3090169131755829, 0.7557617425918579, 0.3333335518836975, 0.755761444568634, -0.5636606812477112;
    pose[23]<<-0.7453562617301941, 0.5773501992225647, 0.33333301544189453, -0.5773502588272095, -0.8090170621871948, 0.11026358604431152, 0.33333277702331543, -0.11026426404714584, 0.9363391995429993;
    pose[24]<<0.33333346247673035, 0.9341723322868347, 0.12732207775115967, 0.35682255029678345, -1.5675064446440956e-07, -0.9341722130775452, -0.872677743434906, 0.35682228207588196, -0.3333338499069214;
    pose[25]<<0.33333346247673035, 0.9341723322868347, 0.12732207775115967, 0.5773497819900513, -0.30901703238487244, 0.7557617425918579, 0.7453562617301941, -0.1784113049507141, -0.642349898815155;
    pose[26]<<0.33333346247673035, 0.9341723322868347, 0.12732207775115967, -0.9341723918914795, 0.30901724100112915, 0.17841041088104248, 0.12732136249542236, -0.17841094732284546, 0.9756838083267212;
    pose[27]<<1.0, 0.0, 0.0, 0.0, -0.49999991059303284, -0.8660255074501038, 0.0, 0.8660255074501038, -0.49999991059303284;
    pose[28]<<1.0, 0.0, 0.0, 0.0, -0.5000000596046448, 0.866025447845459, 0.0, -0.8660253882408142, -0.5000000596046448;
    pose[29]<<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    pose[30]<<0.7453562617301941, -0.5773501992225647, -0.33333301544189453, -0.5773502588272095, -0.8090170621871948, 0.11026366800069809, -0.3333328366279602, 0.11026419699192047, -0.9363391995429993;
    pose[31]<<0.7453562617301941, -0.5773501992225647, -0.33333301544189453, 0.5773499011993408, 0.30901700258255005, 0.7557616829872131, -0.3333335518836975, -0.755761444568634, 0.5636608004570007;
    pose[32]<<0.7453562617301941, -0.5773501992225647, -0.33333301544189453, 3.427267074584961e-07, 0.5000000596046448, -0.8660253882408142, 0.6666664481163025, 0.6454972624778748, 0.37267836928367615;
    pose[33]<<-0.33333346247673035, -0.9341723322868347, -0.12732207775115967, -0.9341723918914795, 0.30901721119880676, 0.17841050028800964, -0.12732143700122833, 0.17841097712516785, -0.9756838083267212;
    pose[34]<<-0.33333346247673035, -0.9341723322868347, -0.12732207775115967, 0.577349841594696, -0.3090170621871948, 0.7557616829872131, -0.7453563213348389, 0.17841126024723053, 0.6423499584197998;
    pose[35]<<-0.33333346247673035, -0.9341723322868347, -0.12732207775115967, 0.35682249069213867, -1.418494832705619e-07, -0.9341722130775452, 0.872677743434906, -0.35682228207588196, 0.3333337903022766;
    pose[36]<<-1.0, 0.0, 0.0, 0.0, 1.0, 8.742277657347586e-08, 0.0, 8.742277657347586e-08, -1.0;
    pose[37]<<-1.0, 0.0, 0.0, 0.0, -0.5000001192092896, 0.8660253882408142, -7.450580596923828e-09, 0.8660253286361694, 0.5000001192092896;
    pose[38]<<-1.0, 0.0, 0.0, 0.0, -0.49999985098838806, -0.8660255074501038, 0.0, -0.8660255074501038, 0.49999985098838806;
    pose[39]<<-0.33333346247673035, 0.9341723322868347, -0.12732192873954773, 0.9341723918914795, 0.30901727080345154, -0.17841044068336487, -0.12732143700122833, -0.17841079831123352, -0.9756838083267212;
    pose[40]<<-0.33333346247673035, 0.9341723322868347, -0.12732192873954773, -0.356822669506073, -3.647899973202584e-07, 0.9341722130775452, 0.8726776838302612, 0.3568222224712372, 0.33333396911621094;
    pose[41]<<-0.33333346247673035, 0.9341723322868347, -0.12732192873954773, -0.5773497223854065, -0.3090168833732605, -0.7557618021965027, -0.7453562617301941, -0.17841143906116486, 0.6423497796058655;
    pose[42]<<0.7453562617301941, 0.5773501992225647, -0.33333292603492737, 0.5773502588272095, -0.8090170621871948, -0.11026380211114883, -0.3333328366279602, -0.11026403307914734, -0.9363391995429993;
    pose[43]<<0.7453562617301941, 0.5773501992225647, -0.33333292603492737, -4.6193599700927734e-07, 0.5000000596046448, 0.866025447845459, 0.6666664481163025, -0.6454972624778748, 0.37267839908599854;
    pose[44]<<0.7453562617301941, 0.5773501992225647, -0.33333292603492737, -0.577349841594696, 0.30901700258255005, -0.7557616829872131, -0.3333336114883423, 0.7557613253593445, 0.563660740852356;
    pose[45]<<0.33333367109298706, -0.35682183504104614, -0.8726779818534851, -0.9341722726821899, 1.2060137066782772e-07, -0.3568224310874939, 0.12732207775115967, 0.9341724514961243, -0.33333301544189453;
    pose[46]<<0.33333367109298706, -0.35682183504104614, -0.8726779818534851, 0.35682204365730286, -0.8090171217918396, 0.4670860767364502, -0.8726778626441956, -0.46708622574806213, -0.14235073328018188;
    pose[47]<<0.33333367109298706, -0.35682183504104614, -0.8726779818534851, 0.5773501992225647, 0.8090171217918396, -0.11026371270418167, 0.7453559637069702, -0.46708622574806213, 0.4756837785243988;
    pose[48]<<-0.33333295583724976, -0.5773497819900513, -0.7453565001487732, -0.3568214476108551, 0.8090173602104187, -0.4670860767364502, 0.8726783990859985, 0.11026395112276077, -0.47568291425704956;
    pose[49]<<-0.33333295583724976, -0.5773497819900513, -0.7453565001487732, -0.5773508548736572, -0.5000001788139343, 0.6454965472221375, -0.7453556656837463, 0.6454975605010986, -0.1666668802499771;
    pose[50]<<-0.33333295583724976, -0.5773497819900513, -0.7453565001487732, 0.9341723918914795, -0.309017151594162, -0.17841054499149323, -0.12732264399528503, -0.7557615637779236, 0.6423498392105103;
    pose[51]<<-0.7453553080558777, 5.828192684020905e-08, -0.6666674613952637, 0.5773510336875916, 0.5, -0.6454966068267822, 0.33333373069763184, -0.866025447845459, -0.37267768383026123;
    pose[52]<<-0.7453553080558777, 5.828192684020905e-08, -0.6666674613952637, -0.5773510336875916, 0.49999991059303284, 0.6454967260360718, 0.33333373069763184, 0.866025447845459, -0.3726775348186493;
    pose[53]<<-0.7453553080558777, 5.828192684020905e-08, -0.6666674613952637, 5.828192684020905e-08, -1.0, -1.525838086990916e-07, -0.6666674613952637, -1.525838086990916e-07, 0.7453553080558777;
    pose[54]<<-0.33333295583724976, 0.5773499011993408, -0.7453563809394836, 0.5773508548736572, -0.5000000596046448, -0.645496666431427, -0.7453556656837463, -0.6454975605010986, -0.16666699945926666;
    pose[55]<<-0.33333295583724976, 0.5773499011993408, -0.7453563809394836, 0.3568214476108551, 0.8090172410011292, 0.46708622574806213, 0.8726783990859985, -0.1102638766169548, -0.47568297386169434;
    pose[56]<<-0.33333295583724976, 0.5773499011993408, -0.7453563809394836, -0.9341723918914795, -0.3090173304080963, 0.1784103959798813, -0.12732282280921936, 0.755761444568634, 0.6423500180244446;
    pose[57]<<0.33333367109298706, 0.3568219542503357, -0.8726779222488403, -0.35682204365730286, -0.80901700258255, -0.46708622574806213, -0.8726778626441956, 0.4670862853527069, -0.14235064387321472;
    pose[58]<<0.33333367109298706, 0.3568219542503357, -0.8726779222488403, 0.9341722726821899, 5.821256365834415e-08, 0.3568224310874939, 0.12732207775115967, -0.9341724514961243, -0.33333316445350647;
    pose[59]<<0.33333367109298706, 0.3568219542503357, -0.8726779222488403, -0.5773503184318542, 0.80901700258255, 0.11026376485824585, 0.7453558444976807, 0.46708622574806213, 0.47568386793136597;
    return pose;
  }

  static  std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> gen_rand_init_pose(int discrete_rpy_num) {

    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_all;
    rot_all.reserve(discrete_rpy_num * discrete_rpy_num * discrete_rpy_num);
    for (int i = 0; i < discrete_rpy_num; i++) {
      for ( int j = 0; j < discrete_rpy_num; j++) {
        for (int k = 0; k < discrete_rpy_num; k++) {
          
          Eigen::Matrix3f rot;
        
          rot = Eigen::AngleAxisf( i / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf( j / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( k / (float)discrete_rpy_num * M_PI, Eigen::Vector3f::UnitZ());
          rot_all.emplace_back(rot);
        //std::cout<<"push "<<rot<<"\n";
        }
      }
    }
    return rot_all;
  }
  


  Eigen::Matrix4f CvoGPU::get_nearest_init_pose(const CvoPointCloud& source,
                                                const CvoPointCloud& target,
                                                float init_guess_ell,
                                                int num_guess,
                                                double * times) const {
    
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> init_rots =  gen_icosahedron_init_rots(); //gen_rand_init_pose(num_guess);
    
    auto start = std::chrono::system_clock::now();
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();  // from source frame to the target frame
    float curr_max_ip = 0;
    for (auto && init_rot : init_rots) {
      Eigen::Matrix4f tmp_init_guess =  Eigen::Matrix4f::Identity();
      tmp_init_guess.block<3,3>(0,0) = init_rot;

      Eigen::Matrix4f init_inv = tmp_init_guess.inverse();
      float ip = this->function_angle(source, target, init_inv, init_guess_ell);
      std::cout<<"inner product of init guess "<<tmp_init_guess<<" is "<<ip<<"\n";
      if (ip > curr_max_ip) {
        curr_max_ip = ip;
        init_guess = tmp_init_guess;
      }
    }
    auto end = std::chrono::system_clock::now();
    //double elapsed =
    if (times)
      *times += std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    return init_guess;
    //std::cout<<"Chosen init guess is "<<init_guess<<", with inner product "<<curr_max_ip<<", the init search takes "<<elapsed<< "ms\n";
    
  }




}
