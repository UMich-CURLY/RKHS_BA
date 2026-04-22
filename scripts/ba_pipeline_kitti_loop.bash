#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <dataset_base> <params_yaml> <graph_root> <tracking_root> <output_root> [seq ...]"
  exit 1
fi

dataset_base=$1
params_yaml=$2
graph_root=$3
tracking_root=$4
output_root=$5
shift 5

if [[ $# -gt 0 ]]; then
  seqs=("$@")
else
  seqs=(07 05 09 00 08 02)
fi

for seq in "${seqs[@]}"; do
  dataset_folder="${dataset_base}/${seq}"
  graph_file="${graph_root}/${seq}_graph.txt"
  tracking_file="${tracking_root}/${seq}.txt"
  output_prefix="${output_root}/kitti_${seq}"

  echo "Running KITTI BA pipeline for seq ${seq}"
  ./build.bunnytest/main_ba_pipeline_kitti_loop \
    "${dataset_folder}" \
    "${params_yaml}" \
    "${graph_file}" \
    "${output_prefix}" \
    "${tracking_file}"
done
