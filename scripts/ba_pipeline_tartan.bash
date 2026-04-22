#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <dataset_base> <params_yaml> <graph_root> <graph_tag> <output_root> [seq ...]"
  exit 1
fi

dataset_base=$1
params_yaml=$2
graph_root=$3
graph_tag=$4
output_root=$5
shift 5

if [[ $# -gt 0 ]]; then
  seqs=("$@")
else
  seqs=(abandonedfactory)
fi

for seq in "${seqs[@]}"; do
  dataset_folder="${dataset_base}/${seq}/Easy/P001"
  graph_file="${graph_root}/${seq}/${graph_tag}_graph.txt"
  output_prefix="${output_root}/tartan_${seq}_${graph_tag}"

  echo "Running Tartan BA pipeline for seq ${seq}"
  ./build.bunnytest/main_ba_pipeline_tartan \
    "${dataset_folder}" \
    "${params_yaml}" \
    "${graph_file}" \
    "${output_prefix}"
done
