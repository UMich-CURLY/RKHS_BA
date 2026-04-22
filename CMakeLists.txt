cmake_minimum_required(VERSION 3.18)
project(CvoGPUHeaderOnly LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)   # Only CXX, not CUDA

find_library(PCL_IO_LIB pcl_io REQUIRED)
find_library(PCL_COMMON_LIB pcl_common REQUIRED)
find_library(PCL_FILTERS_LIB pcl_filters REQUIRED)
find_library(PCL_SEARCH_LIB pcl_search REQUIRED)
find_library(PCL_KDTREE_LIB pcl_kdtree REQUIRED)
find_library(PCL_SAMPLE_CONSENSUS_LIB pcl_sample_consensus REQUIRED)
find_library(PCL_OCTREE_LIB pcl_octree REQUIRED)
find_library(BOOST_IOSTREAMS_LIB boost_iostreams REQUIRED)
set(CVO_PCL_LIBS
    ${PCL_COMMON_LIB}
    ${PCL_IO_LIB}
    ${PCL_FILTERS_LIB}
    ${PCL_SEARCH_LIB}
    ${PCL_KDTREE_LIB}
    ${PCL_SAMPLE_CONSENSUS_LIB}
    ${PCL_OCTREE_LIB}
    ${BOOST_IOSTREAMS_LIB}
)

# Optional: Boost (some PCL components may need it)
# find_package(Boost COMPONENTS system filesystem)

# ------------------------------------------------------------------------------
# Header-only interface library
# ------------------------------------------------------------------------------
add_library(cvo_header_only INTERFACE)
target_include_directories(cvo_header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/UnifiedCvo>
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cvo_header_only INTERFACE
    Eigen3::Eigen
    CUDA::cublas
    CUDA::cusparse
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    OpenMP::OpenMP_CXX
)
# if(Boost_FOUND)
#     target_link_libraries(cvo_header_only INTERFACE Boost::system Boost::filesystem)
# endif()

# ------------------------------------------------------------------------------
# Instantiation: FEATURE_DIM=3, NUM_CLASS=19
# ------------------------------------------------------------------------------
add_library(cvo_inst_3_19 STATIC
    src/instantiations/CvoGPU_inst_3_19.cu
)
target_link_libraries(cvo_inst_3_19 PUBLIC cvo_header_only)
set_target_properties(cvo_inst_3_19 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------------------------
# Instantiation: FEATURE_DIM=1, NUM_CLASS=19
# ------------------------------------------------------------------------------
add_library(cvo_inst_1_19 STATIC
    src/instantiations/CvoGPU_inst_1_19.cu
)
target_link_libraries(cvo_inst_1_19 PUBLIC cvo_header_only)
set_target_properties(cvo_inst_1_19 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(cvo_dataset_handlers STATIC
    src/dataset_handler/KittiHandler.cpp
    src/dataset_handler/TartanAirHandler.cpp
)
target_link_libraries(cvo_dataset_handlers PUBLIC cvo_header_only ${CVO_PCL_LIBS})

# ------------------------------------------------------------------------------
# Test executables
# ------------------------------------------------------------------------------
# LieGroup test (pure C++)
add_executable(liegroup_test
    src/tests/LieGroup_test.cpp
)
target_link_libraries(liegroup_test PRIVATE cvo_header_only)

add_executable(point_converter_test
    src/tests/PointConverter_test.cpp
)
target_link_libraries(point_converter_test PRIVATE cvo_header_only)

add_executable(point_cloud_loader_test
    src/tests/PointCloudLoader_test.cpp
)
target_link_libraries(point_cloud_loader_test PRIVATE cvo_header_only)
target_link_libraries(point_cloud_loader_test PRIVATE ${CVO_PCL_LIBS})

add_executable(dataset_loader_test
    src/tests/DatasetLoader_test.cpp
)
target_link_libraries(dataset_loader_test PRIVATE cvo_header_only)
target_link_libraries(dataset_loader_test PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(dataset_loader_test PRIVATE cvo_dataset_handlers)

add_executable(ba_pipeline_test
    src/tests/BAPipeline_test.cpp
)
set_source_files_properties(src/tests/BAPipeline_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(ba_pipeline_test PRIVATE cvo_header_only)
target_link_libraries(ba_pipeline_test PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(ba_pipeline_test PRIVATE cvo_dataset_handlers)

add_executable(sparse_kernel_mat64_test
    src/tests/SparseKernelMat64_test.cpp
)
set_source_files_properties(src/tests/SparseKernelMat64_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(sparse_kernel_mat64_test PRIVATE cvo_header_only)

add_executable(sparse_kernel_mat_count_benchmark_test
    src/tests/SparseKernelMatCountBenchmark_test.cpp
)
set_source_files_properties(src/tests/SparseKernelMatCountBenchmark_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(sparse_kernel_mat_count_benchmark_test PRIVATE cvo_header_only)

# CvoGPU test for FEATURE_DIM=3, NUM_CLASS=19
add_executable(cvo_test_3_19
    src/tests/CvoGPU_test.cu
)
target_link_libraries(cvo_test_3_19 PRIVATE cvo_inst_3_19 cvo_header_only)

add_executable(multiframe_irls_test
    src/tests/MultiframeIRLS_test.cpp
)
set_source_files_properties(src/tests/MultiframeIRLS_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(multiframe_irls_test PRIVATE cvo_inst_3_19 cvo_header_only)

add_executable(multiframe_irls_angle_test
    src/tests/MultiframeIRLSAngle_test.cpp
)
set_source_files_properties(src/tests/MultiframeIRLSAngle_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(multiframe_irls_angle_test PRIVATE cvo_inst_3_19 cvo_header_only)

add_executable(kernel_weighting_test
    src/tests/KernelWeighting_test.cpp
)
set_source_files_properties(src/tests/KernelWeighting_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(kernel_weighting_test PRIVATE cvo_inst_3_19 cvo_header_only)

add_executable(multiframe_irls_zero_motion_test
    src/tests/MultiframeIRLSZeroMotion_test.cpp
)
set_source_files_properties(src/tests/MultiframeIRLSZeroMotion_test.cpp PROPERTIES LANGUAGE CUDA)
target_compile_definitions(multiframe_irls_zero_motion_test PRIVATE CVO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(multiframe_irls_zero_motion_test PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(multiframe_irls_zero_motion_test PRIVATE ${CVO_PCL_LIBS})

add_executable(multiframe_bunny_angle_test
    src/tests/MultiframeBunnyAngle_test.cpp
)
set_source_files_properties(src/tests/MultiframeBunnyAngle_test.cpp PROPERTIES LANGUAGE CUDA)
target_compile_definitions(multiframe_bunny_angle_test PRIVATE CVO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(multiframe_bunny_angle_test PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(multiframe_bunny_angle_test PRIVATE ${CVO_PCL_LIBS})

add_executable(multiframe_bunny_test
    src/tests/MultiframeBunny_test.cpp
)
set_source_files_properties(src/tests/MultiframeBunny_test.cpp PROPERTIES LANGUAGE CUDA)
target_compile_definitions(multiframe_bunny_test PRIVATE CVO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(multiframe_bunny_test PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(multiframe_bunny_test PRIVATE ${CVO_PCL_LIBS})

add_executable(multiframe_bunny_trend_test
    src/tests/MultiframeBunnyTrend_test.cpp
)
set_source_files_properties(src/tests/MultiframeBunnyTrend_test.cpp PROPERTIES LANGUAGE CUDA)
target_compile_definitions(multiframe_bunny_trend_test PRIVATE CVO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(multiframe_bunny_trend_test PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(multiframe_bunny_trend_test PRIVATE ${CVO_PCL_LIBS})

add_executable(multiframe_kitti_test
    src/tests/MultiframeKitti_test.cpp
)
set_source_files_properties(src/tests/MultiframeKitti_test.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(multiframe_kitti_test PRIVATE cvo_inst_1_19 cvo_header_only)
target_link_libraries(multiframe_kitti_test PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(multiframe_kitti_test PRIVATE cvo_dataset_handlers)

add_executable(cukdtree_test
    src/tests/CuKdTree_test.cu
)
target_link_libraries(cukdtree_test PRIVATE cvo_header_only)

# Optional: CvoGPU test for FEATURE_DIM=1, NUM_CLASS=19
# add_executable(cvo_test_1_19 src/cvo/CvoGPU_test.cu)
# target_link_libraries(cvo_test_1_19 PRIVATE cvo_inst_1_19 cvo_header_only)

# Multi‑frame bunny test
add_executable(multiframe_bunny
    src/experiments/main_multiframe_bunny.cpp
)
set_source_files_properties(src/experiments/main_multiframe_bunny.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(multiframe_bunny PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(multiframe_bunny PRIVATE ${CVO_PCL_LIBS})

add_executable(main_multi_frame_irls_tartan
    src/experiments/main_multi_frame_irls_tartan.cpp
)
set_source_files_properties(src/experiments/main_multi_frame_irls_tartan.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(main_multi_frame_irls_tartan PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(main_multi_frame_irls_tartan PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(main_multi_frame_irls_tartan PRIVATE cvo_dataset_handlers)

add_executable(main_ba_pipeline_kitti_loop
    src/experiments/main_ba_pipeline_kitti_loop.cpp
)
set_source_files_properties(src/experiments/main_ba_pipeline_kitti_loop.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(main_ba_pipeline_kitti_loop PRIVATE cvo_inst_1_19 cvo_header_only)
target_link_libraries(main_ba_pipeline_kitti_loop PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(main_ba_pipeline_kitti_loop PRIVATE cvo_dataset_handlers)

add_executable(main_ba_pipeline_tartan
    src/experiments/main_ba_pipeline_tartan.cpp
)
set_source_files_properties(src/experiments/main_ba_pipeline_tartan.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(main_ba_pipeline_tartan PRIVATE cvo_inst_3_19 cvo_header_only)
target_link_libraries(main_ba_pipeline_tartan PRIVATE ${CVO_PCL_LIBS})
target_link_libraries(main_ba_pipeline_tartan PRIVATE cvo_dataset_handlers)
