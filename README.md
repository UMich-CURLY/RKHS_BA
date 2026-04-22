# RKHS CVO / RKHS-BA Refactor

This repository is a refactored C++/CUDA implementation of:

- two-frame CVO point cloud registration
- multi-frame RKHS-BA style IRLS registration

Compared with the older `RKHS_BA` tree, this repo pushes most solver logic into templated headers under `include/UnifiedCvo/`, and keeps only small explicit CUDA instantiation libraries for the concrete point types used in the shipped runners and tests.

The current codebase supports:

- geometric kernels
- feature kernels
- semantic-distribution kernels
- pairwise first-order on-manifold CVO alignment on `SE(3)`
- multi-frame IRLS alignment
- CUDA mutual-kNN sparse correlation filling
- dataset-handler-backed KITTI and Tartan runners

Reference papers are in [paper](paper).

## Repository Layout

Main solver code lives under [include/UnifiedCvo](include/UnifiedCvo):

- [include/UnifiedCvo/cvo/CvoGPU.hpp](include/UnifiedCvo/cvo/CvoGPU.hpp)
- [include/UnifiedCvo/cvo/CvoGPU.cuh](include/UnifiedCvo/cvo/CvoGPU.cuh)
- [include/UnifiedCvo/cvo/IRLS_State_GPU.cuh](include/UnifiedCvo/cvo/IRLS_State_GPU.cuh)
- [include/UnifiedCvo/cvo/KernelWeight.hpp](include/UnifiedCvo/cvo/KernelWeight.hpp)
- [include/UnifiedCvo/graph_optimizer/BAPipeline.hpp](include/UnifiedCvo/graph_optimizer/BAPipeline.hpp)
- [include/UnifiedCvo/graph_optimizer/BAPipeline.tpp](include/UnifiedCvo/graph_optimizer/BAPipeline.tpp)

Supporting utilities live under:

- [include/UnifiedCvo/utils](include/UnifiedCvo/utils)
- [include/UnifiedCvo/dataset_handler](include/UnifiedCvo/dataset_handler)

Tests are under:

- [src/tests](src/tests)

Runners are under:

- [src/experiments](src/experiments)

## Design

### Header-heavy, template-first structure

The library target layout is:

- `cvo_header_only`
  - interface target for the templated solver/utilities
- `cvo_inst_3_19`
  - explicit CUDA instantiation for `pcl::PointSemantic<3,19>`
- `cvo_inst_1_19`
  - explicit CUDA instantiation for `pcl::PointSemantic<1,19>`
- `cvo_dataset_handlers`
  - dataset IO and handler-backed runner support

This means:

- most algorithm logic is in headers and `.cuh` files
- users can reuse the templated interfaces directly
- this repo still ships a small number of pre-instantiated libraries for the built-in demos and tests

### Point types

The main point type family is based on [include/UnifiedCvo/utils/PointSemantic.hpp](include/UnifiedCvo/utils/PointSemantic.hpp).

It can store:

- `x, y, z`
- `r, g, b`
- `features[FEATURE_DIM]`
- `label`
- `label_distribution[NUM_CLASS]`
- optional geometric type / normal / covariance data

The generic point cloud wrapper is:

- [include/UnifiedCvo/utils/CvoPointCloud.hpp](include/UnifiedCvo/utils/CvoPointCloud.hpp)

Point conversion is handled by:

- [include/UnifiedCvo/utils/PointConverter.hpp](include/UnifiedCvo/utils/PointConverter.hpp)

Current conversion rules include:

- `PointXYZRGB -> PointSemantic`
  - geometry, RGB, and normalized RGB copied into `features[0:3]`
- `PointXYZI -> PointSemantic`
  - geometry and intensity copied into `features[0]`
- `PointSemantic -> PointSemantic`
  - features and semantic distributions copied directly

## Main Interfaces

### `cvo::CvoGPU<PointT>`

Declared in [include/UnifiedCvo/cvo/CvoGPU.hpp](include/UnifiedCvo/cvo/CvoGPU.hpp).

Core public interface:

```cpp
template <typename PointT>
class CvoGPU {
public:
  using PointCloud = CvoPointCloud<PointT>;

  explicit CvoGPU(const CvoParams& params);

  CvoResultInfo align(const PointCloud& source,
                      const PointCloud& target,
                      const Eigen::Matrix4f& T_init,
                      bool return_association = false) const;

  float function_angle(const PointCloud& source,
                       const PointCloud& target,
                       const Eigen::Matrix4f& T,
                       float ell,
                       bool approximate = true) const;

  void compute_association(const PointCloud& source,
                           const PointCloud& target,
                           const Eigen::Matrix4f& T,
                           float ell,
                           Association& assoc) const;

  int align_multiframe(std::vector<std::shared_ptr<CvoFrameGPU<PointT>>>& frames,
                       const std::vector<std::shared_ptr<BinaryStateGPU<PointT>>>& edge_states,
                       const std::vector<bool>& fixed_flags,
                       double* registration_seconds = nullptr);

  int align_multiframe(const std::vector<CvoPointCloud<PointT>>& clouds,
                       const pgo::MapOfPoses& initial_poses,
                       const pgo::VectorOfConstraints& constraints,
                       pgo::MapOfPoses* optimized_poses = nullptr,
                       double* registration_seconds = nullptr);
};
```

Important convention:

- pairwise alignment uses `T * p_target = p_source`

The current kernel weighting path multiplies:

- spatial/geometric kernel
- feature kernel
- semantic-distribution kernel
- optional geometric-type compatibility

for both pairwise `function_angle()` and multiframe sparse correlation filling.

### `cvo::BAPipeline<PointT>`

Declared in [include/UnifiedCvo/graph_optimizer/BAPipeline.hpp](include/UnifiedCvo/graph_optimizer/BAPipeline.hpp).

Core public interface:

```cpp
struct BAPipelineOptions {
  std::string dataset_type;
  std::filesystem::path dataset_root;
  std::string params_yaml;
  std::filesystem::path graph_file;
  std::filesystem::path output_prefix;
  std::filesystem::path pose_file;
  std::filesystem::path calibration_file;
};

template <typename PointT>
class BAPipeline {
public:
  explicit BAPipeline(BAPipelineOptions options);
  int run();
  static BAGraphSpec read_graph_spec(const std::filesystem::path& path);
};
```

The current pipeline is intentionally thin. It handles:

- graph loading
- dataset handler creation
- point cloud loading for selected frames
- initial pose loading
- simple constraint assembly
- multiframe IRLS solve
- export of trajectory and stacked point clouds

## Parameters

Parameters are defined in [include/UnifiedCvo/cvo/CvoParams.hpp](include/UnifiedCvo/cvo/CvoParams.hpp) and typically loaded with `read_CvoParams_yaml(...)`.

### Pairwise kernel parameters

- `ell_init`, `ell_min`, `ell_max`
  - spatial kernel length-scale schedule
- `sigma`
  - spatial kernel signal scale
- `sp_thres`
  - sparse correlation cutoff
- `c_ell`, `c_sigma`
  - feature kernel length-scale and signal scale
- `s_ell`, `s_sigma`
  - semantic-distribution kernel length-scale and signal scale

### Pairwise solver parameters

- `MAX_ITER`
  - maximum pairwise iterations
- `eps`, `eps_2`
  - stopping thresholds
- `min_step`, `max_step`
  - pairwise step-size bounds
- `ell_decay_rate`, `ell_decay_start`
  - pairwise `ell` schedule
- `nearest_neighbors_max`
  - sparse fill neighbor cap

### Pairwise mode flags

- `is_using_geometry`
  - enable spatial/geometric kernel
- `is_using_intensity`
  - enable feature kernel over `features[]`
- `is_using_semantics`
  - enable semantic-distribution kernel over `label_distribution[]`
- `is_using_geometric_type`
  - multiply by geometric-type compatibility
- `is_using_range_ell`
  - enable range-adaptive spatial `ell`
- `is_using_kdtree`
  - enable CUDA kd-tree accelerated sparse search
- `is_global_angle_registration`
  - enable global rotation search when used by a caller

### Multi-frame IRLS parameters

- `multiframe_max_iters`
  - global maximum IRLS iterations
- `multiframe_ell_init`
  - initial multiframe kernel length-scale
- `multiframe_ell_min`
  - minimum multiframe `ell`
- `multiframe_ell_decay_rate`
  - decay ratio once the current `ell` level converges
- `multiframe_iterations_per_ell`
  - inner iteration budget at each `ell`
- `multiframe_num_neighbors`
  - sparse neighbors per edge
- `multiframe_min_nonzeros`
  - minimum sparse support required before solve
- `multiframe_sparse_fill_backend`
  - sparse fill backend
  - `0 = cpu_mutual`
  - `1 = gpu_mutual_knn`

### Logging/debug parameters

- `multiframe_enable_iteration_log`
  - write per-iteration objective CSV
- `multiframe_iteration_log_path`
  - path for objective trace CSV
- `multiframe_enable_pose_log`
  - write per-iteration pose CSV
- `multiframe_pose_log_path`
  - path for pose trace CSV

## Build

```bash
cmake -S . -B build
cmake --build build -j1
```

For repeated local work, this repo has usually been built into `build.bunnytest`:

```bash
cmake -S . -B build.bunnytest
cmake --build build.bunnytest -j1
```

Important targets:

- `cvo_test_3_19`
- `multiframe_irls_test`
- `multiframe_irls_angle_test`
- `multiframe_irls_zero_motion_test`
- `kernel_weighting_test`
- `cukdtree_test`
- `dataset_loader_test`
- `ba_pipeline_test`
- `main_multi_frame_irls_tartan`
- `main_ba_pipeline_kitti_loop`
- `main_ba_pipeline_tartan`

## Demos

### Two-frame CVO Demo

The simplest built-in two-frame demo is:

```bash
./build.bunnytest/cvo_test_3_19
```

That test:

- creates two synthetic semantic point clouds
- runs one pairwise CVO step
- checks that the function angle improves

If you want to call the library directly:

```cpp
using PointT = pcl::PointSemantic<3, 19>;
using Cloud = cvo::CvoPointCloud<PointT>;

cvo::CvoParams params;
params.is_using_geometry = 1;
params.is_using_intensity = 1;
params.is_using_semantics = 1;

cvo::CvoGPU<PointT> solver(params);
Cloud source, target;
Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();

cvo::CvoResultInfo result = solver.align(source, target, T_init, true);
```

### Two-frame and Multi-frame Regression Demos

The fastest built-in smoke tests are:

```bash
./build.bunnytest/multiframe_irls_zero_motion_test
./build.bunnytest/multiframe_bunny_test 2
./build.bunnytest/multiframe_bunny_test 4
```

Useful variants of `multiframe_bunny_test`:

```bash
./build.bunnytest/multiframe_bunny_test \
  <num_frames> \
  <sparse_fill_backend:0=cpu,1=gpu> \
  <linear_backend:0=cpu_dense,1=gpu_dense,2=cpu_sparse,3=gpu_sparse> \
  <objective_backend:0=cpu,1=gpu> \
  <enable_line_search:0/1> \
  [release_binary_state_gpu_each_iter:0/1] \
  [stream_frame_clouds_gpu:0/1]
```

Example:

```bash
./build.bunnytest/multiframe_bunny_test 2 1 3 0 1 1 1
```

This runs:

- 2-frame bunny registration
- GPU mutual-kNN sparse fill
- GPU sparse linear backend
- CPU objective evaluation
- line search enabled
- early binary-state GPU release enabled
- streamed frame-cloud GPU residency enabled

Outputs are written in the repo root:

- `multiframe_bunny_init_stack_<N>.ply`
- `multiframe_bunny_final_stack_<N>.ply`
- `multiframe_bunny_trace_<N>.csv`
- `multiframe_bunny_pose_trace_<N>.csv`

### Generic Multi-frame Runner

### Generic multiframe runner

The generic runner is:

- [src/experiments/main_multi_frame_irls_tartan.cpp](src/experiments/main_multi_frame_irls_tartan.cpp)

Usage:

```bash
./build.bunnytest/main_multi_frame_irls_tartan \
  <dataset_type:{pcd|kitti_lidar|tartan_rgbd}> \
  <dataset_root> \
  <params_yaml> \
  <graph_file> \
  <output_prefix> \
  [pose_file] [calibration_file]
```

Outputs:

- `<output_prefix>_trajectory.txt`
- `<output_prefix>_stacked_init.ply`
- `<output_prefix>_stacked_final.ply`

This runner is useful for:

- direct multiframe IRLS on a selected graph
- dataset-backed small tests
- debugging without the extra BA pipeline layer

### Dedicated BA Pipeline Runners

KITTI loop-style runner:

```bash
./build.bunnytest/main_ba_pipeline_kitti_loop \
  <kitti_sequence_root> \
  <params_yaml> \
  <graph_file> \
  <output_prefix> \
  <tracking_pose_file> \
  [calibration_file]
```

Tartan runner:

```bash
./build.bunnytest/main_ba_pipeline_tartan \
  <tartan_traj_root> \
  <params_yaml> \
  <graph_file> \
  <output_prefix> \
  [pose_file] [calibration_file]
```

### Recommended Shell Scripts

Current helper scripts:

- [scripts/ba_pipeline_kitti_loop.bash](scripts/ba_pipeline_kitti_loop.bash)
- [scripts/ba_pipeline_tartan.bash](scripts/ba_pipeline_tartan.bash)
- [scripts/run_kitti_gpu_sparse_ba_eval.bash](scripts/run_kitti_gpu_sparse_ba_eval.bash)

Example KITTI run:

```bash
bash scripts/ba_pipeline_kitti_loop.bash \
  /run/media/rayzhang/Samsung_T5/kitti_lidar/dataset/sequences \
  ../RKHS_BA/cvo_params/cvo_irls_kitti_ba_params.yaml \
  /tmp/cvo_runner_tests \
  ../RKHS_BA/results/mulls_with_loop \
  /tmp/cvo_runner_outputs \
  07
```

Example Tartan run:

```bash
bash scripts/ba_pipeline_tartan.bash \
  /run/media/rayzhang/Samsung_T5/tartanair \
  ../RKHS_BA/cvo_params/cvo_tartan_demo_params.yaml \
  /tmp/cvo_runner_tests \
  small \
  /tmp/cvo_runner_outputs \
  abandonedfactory
```

### KITTI BA + Evaluation Demo

The current end-to-end KITTI launcher is:

- [scripts/run_kitti_gpu_sparse_ba_eval.bash](scripts/run_kitti_gpu_sparse_ba_eval.bash)

It does all of the following:

- builds `main_ba_pipeline_kitti_loop`
- builds the KITTI odometry devkit evaluator from `../RKHS_BA/devkit/cpp/evaluate_odometry.cpp`
- builds a radius graph from the input tracking trajectory
- writes a temporary IRLS YAML
- runs BA
- evaluates both init and BA trajectories against KITTI `poses.txt`

Usage:

```bash
./scripts/run_kitti_gpu_sparse_ba_eval.bash \
  <seq> \
  [dataset_base] \
  [tracking_root] \
  [output_root] \
  [linear_backend] \
  [release_binary_state_each_iter] \
  [stream_frame_clouds_gpu]
```

Important arguments:

- `linear_backend`
  - `2 = cpu_sparse_block`
  - `3 = gpu_sparse_block`
- `release_binary_state_each_iter`
  - `1` frees per-edge GPU buffers during sparse streamed assembly
- `stream_frame_clouds_gpu`
  - `1` uploads/frees frame clouds on demand instead of keeping all frames resident on GPU

Recommended large-KITTI memory-saving launch:

```bash
./scripts/run_kitti_gpu_sparse_ba_eval.bash \
  05 \
  /run/media/rayzhang/Samsung_T5/kitti_lidar/dataset/sequences \
  ../RKHS_BA/results/mulls_with_loop \
  /tmp/cvo_runner_tests \
  2 \
  1 \
  1
```

Outputs are written under:

- `/tmp/cvo_runner_tests/kitti_<seq>_<backend_tag>/`

including:

- `<seq>_<backend_tag>_irls.yaml`
- `<seq>_radius2m_graph.txt`
- `kitti_<seq>_<backend_tag>_trajectory.txt`
- `kitti_<seq>_<backend_tag>_stacked_init.ply`
- `kitti_<seq>_<backend_tag>_stacked_final.ply`
- `init_eval.txt`
- `ba_eval.txt`

## Testing

Useful solver regression tests:

```bash
./build.bunnytest/kernel_weighting_test
./build.bunnytest/multiframe_irls_angle_test
./build.bunnytest/multiframe_irls_zero_motion_test
./build.bunnytest/multiframe_bunny_test 2
./build.bunnytest/multiframe_bunny_test 4
./build.bunnytest/cukdtree_test
```

For multiframe trend traces:

```bash
python3 scripts/check_multiframe_trace.py <trace.csv>
python3 scripts/check_bunny_pose_trace.py <pose_trace.csv>
python3 scripts/check_bunny_multistep_trace.py <trace.csv> <pose_trace.csv>
```

## Current Limitations

- `BAPipeline` is still a thin orchestration layer, not yet a full feature-parity port of the old monolithic `RKHS_BA` runners
- current `BAPipeline` constraints are still mainly used as an edge list for multiframe IRLS; it is not yet a full odometry/loop-closure weighted pose-graph BA formulation
- the repo is header-heavy, but not purely header-only because the shipped CUDA instantiation libraries remain necessary for the built-in point types
- many older experiments and utilities from `RKHS_BA` have not yet been ported here

## Relevant Files

- solver core:
  - [include/UnifiedCvo/cvo/CvoGPU.cuh](include/UnifiedCvo/cvo/CvoGPU.cuh)
- sparse multiframe edge state:
  - [include/UnifiedCvo/cvo/IRLS_State_GPU.cuh](include/UnifiedCvo/cvo/IRLS_State_GPU.cuh)
- kernel weighting:
  - [include/UnifiedCvo/cvo/KernelWeight.hpp](include/UnifiedCvo/cvo/KernelWeight.hpp)
- BA pipeline:
  - [include/UnifiedCvo/graph_optimizer/BAPipeline.hpp](include/UnifiedCvo/graph_optimizer/BAPipeline.hpp)
- dataset-handler glue:
  - [include/UnifiedCvo/utils/DatasetHandlerUtils.hpp](include/UnifiedCvo/utils/DatasetHandlerUtils.hpp)
- tests:
  - [src/tests](src/tests)
