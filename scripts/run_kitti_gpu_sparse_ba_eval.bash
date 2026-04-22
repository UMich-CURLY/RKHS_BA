#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 7 ]]; then
  echo "Usage: $0 <seq> [dataset_base] [tracking_root] [output_root] [linear_backend] [release_binary_state_each_iter] [stream_frame_clouds_gpu]"
  echo "linear_backend: 2=cpu_sparse_block, 3=gpu_sparse_block (default)"
  echo "release_binary_state_each_iter: 0=keep allocated (default), 1=free/realloc each IRLS iter"
  echo "stream_frame_clouds_gpu: 0=keep all frame clouds resident, 1=upload/free frame clouds on demand"
  echo "Example: $0 05 /run/media/rayzhang/Samsung_T5/kitti_lidar/dataset/sequences ../RKHS_BA/results/mulls_with_loop /tmp/cvo_runner_tests 2 1 1"
  exit 1
fi

seq=$1
dataset_base=${2:-/run/media/rayzhang/Samsung_T5/kitti_lidar/dataset/sequences}
tracking_root=${3:-../RKHS_BA/results/mulls_with_loop}
output_root=${4:-/tmp/cvo_runner_tests}
linear_backend=${5:-3}
release_binary_state_each_iter=${6:-1}
stream_frame_clouds_gpu=${7:-1}
objective_backend=1
if [[ "${release_binary_state_each_iter}" == "1" || "${stream_frame_clouds_gpu}" == "1" ]]; then
  objective_backend=0
fi

dataset_root="${dataset_base}/${seq}"
tracking_pose_file="${tracking_root}/${seq}/odom${seq}_kitti.txt"
gt_pose_file="${dataset_root}/poses.txt"
backend_tag="gpu_sparse"
if [[ "${linear_backend}" == "2" ]]; then
  backend_tag="cpu_sparse"
fi
work_dir="${output_root}/kitti_${seq}_${backend_tag}"
graph_file="${work_dir}/${seq}_radius2m_graph.txt"
params_yaml="${work_dir}/${seq}_${backend_tag}_irls.yaml"
output_prefix="${work_dir}/kitti_${seq}_${backend_tag}"
eval_root="${work_dir}/eval"
eval_bin="${work_dir}/evaluate_odometry"

mkdir -p "${work_dir}" "${eval_root}/${seq}"

if [[ ! -d "${dataset_root}" ]]; then
  echo "Missing dataset root: ${dataset_root}" >&2
  exit 1
fi
if [[ ! -f "${tracking_pose_file}" ]]; then
  echo "Missing tracking pose file: ${tracking_pose_file}" >&2
  exit 1
fi
if [[ ! -f "${gt_pose_file}" ]]; then
  echo "Missing KITTI GT pose file: ${gt_pose_file}" >&2
  exit 1
fi

echo "[1/5] Building KITTI BA runner"
cmake --build build.bunnytest --target main_ba_pipeline_kitti_loop -j1

echo "[2/5] Building KITTI odometry evaluator"
g++ -O2 -o "${eval_bin}" \
  ../RKHS_BA/devkit/cpp/evaluate_odometry.cpp \
  ../RKHS_BA/devkit/cpp/matrix.cpp

echo "[3/5] Generating radius graph and params"
python3 - <<PY
from pathlib import Path

seq = "${seq}"
traj_path = Path("${tracking_pose_file}")
graph_path = Path("${graph_file}")
yaml_path = Path("${params_yaml}")
work_dir = Path("${work_dir}")

poses = []
with traj_path.open() as f:
    for line in f:
        vals = [float(x) for x in line.strip().split()]
        if len(vals) == 13:
            vals = vals[1:]
        if len(vals) != 12:
            continue
        poses.append((vals[3], vals[7], vals[11]))

edges = []
for i, (xi, yi, zi) in enumerate(poses):
    for j in range(i + 1, len(poses)):
        xj, yj, zj = poses[j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        if dx * dx + dy * dy + dz * dz <= 4.0:
            edges.append((i, j))

with graph_path.open("w") as f:
    f.write(f"{len(poses)} {len(edges)}\\n")
    f.write(" ".join(str(i) for i in range(len(poses))) + "\\n")
    for i, j in edges:
        f.write(f"{i} {j}\\n")

with yaml_path.open("w") as f:
    f.write("\\n".join([
        "sigma: 0.1",
        "sp_thres: 0.001",
        "c_ell: 0.1",
        "c_sigma: 1.0",
        "is_using_geometry: 1",
        "is_using_intensity: 1",
        "is_using_semantics: 0",
        "is_using_geometric_type: 0",
        "is_using_kdtree: 0",
        "multiframe_max_iters: 10",
        "multiframe_ell_init: 1.0",
        "multiframe_ell_min: 0.5",
        "multiframe_ell_decay_rate: 0.8",
        "multiframe_iterations_per_ell: 3",
        "multiframe_downsample_voxel_size: 1.0",
        "multiframe_num_neighbors: 16",
        "multiframe_min_nonzeros: 100",
        "multiframe_sparse_fill_backend: 1",
        "multiframe_linear_system_backend: ${linear_backend}",
        "multiframe_objective_eval_backend: ${objective_backend}",
        "multiframe_enable_line_search: 1",
        "multiframe_release_binary_state_gpu_each_iter: ${release_binary_state_each_iter}",
        "multiframe_stream_frame_clouds_gpu: ${stream_frame_clouds_gpu}",
        f"multiframe_enable_iteration_log: 1",
        f"multiframe_iteration_log_path: {work_dir / (seq + '_trace.csv')}",
        f"multiframe_enable_pose_log: 1",
        f"multiframe_pose_log_path: {work_dir / (seq + '_pose_trace.csv')}",
        ""
    ]))

print(f"frames={len(poses)} edges={len(edges)}")
print(graph_path)
print(yaml_path)
PY

echo "[4/5] Running GPU sparse BA"
TIMEFORMAT='elapsed=%3R'
time ./build.bunnytest/main_ba_pipeline_kitti_loop \
  "${dataset_root}" \
  "${params_yaml}" \
  "${graph_file}" \
  "${output_prefix}" \
  "${tracking_pose_file}"

ba_traj="${output_prefix}_trajectory.txt"
if [[ ! -f "${ba_traj}" ]]; then
  echo "BA trajectory not produced: ${ba_traj}" >&2
  exit 1
fi
ba_eval_traj="${eval_root}/${seq}/ba.txt"
python3 - <<PY
from pathlib import Path
src = Path("${ba_traj}")
dst = Path("${ba_eval_traj}")
lines = []
for line in src.read_text().splitlines():
    parts = line.split()
    if len(parts) >= 13:
        parts = parts[1:13]
    lines.append(" ".join(parts))
dst.write_text("\\n".join(lines) + "\\n")
PY

echo "[5/5] Evaluating init and BA against ${gt_pose_file}"
cp "${gt_pose_file}" "${eval_root}/${seq}/poses.txt"
cp "${tracking_pose_file}" "${eval_root}/${seq}/init.txt"

"${eval_bin}" "${seq#0}" "${eval_root}/" poses.txt "${eval_root}/" init.txt | tee "${work_dir}/init_eval.txt"
"${eval_bin}" "${seq#0}" "${eval_root}/" poses.txt "${eval_root}/" ba.txt | tee "${work_dir}/ba_eval.txt"

echo "Artifacts:"
echo "  graph: ${graph_file}"
echo "  params: ${params_yaml}"
echo "  ba trajectory: ${ba_traj}"
echo "  init eval: ${work_dir}/init_eval.txt"
echo "  ba eval: ${work_dir}/ba_eval.txt"
echo "  release_binary_state_each_iter: ${release_binary_state_each_iter}"
echo "  stream_frame_clouds_gpu: ${stream_frame_clouds_gpu}"
