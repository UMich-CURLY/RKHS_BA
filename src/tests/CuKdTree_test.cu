#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cukdtree/cukdtree.cuh"
#include "utils/PointSemantic.hpp"

using PointType = pcl::PointSemantic<3, 19>;

namespace {

PointType makePoint(float x, float y, float z) {
  PointType p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

std::vector<PointType> makeStructuredCloud(int n) {
  std::vector<PointType> points;
  points.reserve(n);
  for (int i = 0; i < n; ++i) {
    const float x = static_cast<float>(i);
    const float y = static_cast<float>((i % 7) - 3) * 0.25f;
    const float z = static_cast<float>((i % 5) - 2) * 0.15f;
    points.push_back(makePoint(x, y, z));
  }
  return points;
}

float sqDist(const PointType& a, const PointType& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

void verify_tree_layout(const thrust::host_vector<perl_registration::TreeNode>& tree, int n_points) {
  std::vector<int> used_indices;
  used_indices.reserve(tree.size());
  std::vector<int> used_points;
  used_points.reserve(n_points);

  for (int i = 0; i < static_cast<int>(tree.size()); ++i) {
    if (tree[i].point >= 0) {
      used_indices.push_back(i);
      used_points.push_back(tree[i].point);
    }
  }

  assert(static_cast<int>(used_indices.size()) == n_points);
  for (int i = 0; i < n_points; ++i) {
    assert(used_indices[i] == i);
  }

  std::sort(used_points.begin(), used_points.end());
  for (int i = 0; i < n_points; ++i) {
    assert(used_points[i] == i);
  }

  for (int i = 0; i < n_points; ++i) {
    const auto& node = tree[i];
    if (node.leftChild >= 0) {
      assert(node.leftChild < n_points);
    }
    if (node.rightChild >= 0) {
      assert(node.rightChild < n_points);
    }
  }
}

std::vector<PointType> brute_force_knn(const std::vector<PointType>& cloud, const PointType& query, int k) {
  std::vector<int> indices(cloud.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int lhs, int rhs) { return sqDist(cloud[lhs], query) < sqDist(cloud[rhs], query); });
  std::vector<PointType> neighbors;
  neighbors.reserve(k);
  for (int i = 0; i < k; ++i) {
    neighbors.push_back(cloud[indices[i]]);
  }
  return neighbors;
}

auto point_key(const PointType& p) {
  return std::make_tuple(p.x, p.y, p.z);
}

std::vector<float> sorted_distances(const std::vector<PointType>& points, const PointType& query) {
  std::vector<float> dists;
  dists.reserve(points.size());
  for (const auto& point : points) {
    dists.push_back(sqDist(point, query));
  }
  std::sort(dists.begin(), dists.end());
  return dists;
}

void verify_knn_matches(const std::vector<PointType>& cloud, int k) {
  thrust::device_vector<PointType> device_cloud(cloud.begin(), cloud.end());
  perl_registration::cuKdTree<PointType> tree;
  tree.SetInputCloud(device_cloud);
  thrust::host_vector<PointType> host_sorted_cloud = device_cloud;

  thrust::host_vector<perl_registration::TreeNode> host_tree = tree.tree();
  verify_tree_layout(host_tree, static_cast<int>(cloud.size()));

  std::vector<PointType> queries = {
      makePoint(0.1f, 0.0f, 0.0f),
      makePoint(static_cast<float>(cloud.size()) * 0.5f, 0.2f, -0.1f),
      makePoint(static_cast<float>(cloud.size()) - 1.2f, -0.2f, 0.25f)};
  thrust::device_vector<PointType> device_queries(queries.begin(), queries.end());

  thrust::device_vector<int> device_indices;
  tree.NearestKSearch(device_queries, k, device_indices);
  thrust::host_vector<int> host_indices = device_indices;

  for (int qi = 0; qi < static_cast<int>(queries.size()); ++qi) {
    std::vector<PointType> expected = brute_force_knn(cloud, queries[qi], k);
    std::vector<PointType> actual;
    actual.reserve(k);
    for (int j = 0; j < k; ++j) {
      actual.push_back(host_sorted_cloud[host_indices[qi * k + j]]);
    }

    std::sort(expected.begin(), expected.end(),
              [&](const PointType& lhs, const PointType& rhs) { return point_key(lhs) < point_key(rhs); });
    std::sort(actual.begin(), actual.end(),
              [&](const PointType& lhs, const PointType& rhs) { return point_key(lhs) < point_key(rhs); });

    const std::vector<float> expected_dists = sorted_distances(expected, queries[qi]);
    const std::vector<float> actual_dists = sorted_distances(actual, queries[qi]);
    assert(expected_dists.size() == actual_dists.size());
    for (int j = 0; j < k; ++j) {
      assert(std::fabs(expected_dists[j] - actual_dists[j]) < 1e-5f);
    }
  }
}

std::vector<PointType> makeDuplicateCloud() {
  return {
      makePoint(0.0f, 0.0f, 0.0f),
      makePoint(0.0f, 0.0f, 0.0f),
      makePoint(1.0f, 0.0f, 0.0f),
      makePoint(1.0f, 0.0f, 0.0f),
      makePoint(0.0f, 1.0f, 0.0f),
      makePoint(0.0f, 1.0f, 0.0f),
      makePoint(0.0f, 0.0f, 1.0f),
      makePoint(0.0f, 0.0f, 1.0f)};
}

std::vector<PointType> makeSymmetricCloud() {
  return {
      makePoint(-1.0f, 0.0f, 0.0f),
      makePoint(1.0f, 0.0f, 0.0f),
      makePoint(0.0f, -1.0f, 0.0f),
      makePoint(0.0f, 1.0f, 0.0f),
      makePoint(0.0f, 0.0f, -1.0f),
      makePoint(0.0f, 0.0f, 1.0f)};
}

void verify_case_suite(const std::vector<int>& sizes, int requested_k) {
  for (int n : sizes) {
    const int k = std::min(requested_k, n);
    verify_knn_matches(makeStructuredCloud(n), k);
  }
}

void verify_permutation_invariance(int n, int k) {
  std::vector<PointType> cloud = makeStructuredCloud(n);
  std::mt19937 rng(7);
  std::shuffle(cloud.begin(), cloud.end(), rng);
  verify_knn_matches(cloud, k);
}

void verify_special_clouds() {
  verify_knn_matches(makeDuplicateCloud(), 4);
  verify_knn_matches(makeSymmetricCloud(), 4);
}

void verify_k_boundaries() {
  const std::vector<int> sizes = {1, 2, 5, 17, 33, cvo::KDTREE_K_SIZE - 1, cvo::KDTREE_K_SIZE, cvo::KDTREE_K_SIZE + 1};
  for (int n : sizes) {
    const std::vector<PointType> cloud = makeStructuredCloud(n);
    verify_knn_matches(cloud, 1);
    verify_knn_matches(cloud, std::min(3, n));
    verify_knn_matches(cloud, std::min(cvo::KDTREE_K_SIZE, n));
    verify_knn_matches(cloud, n);
  }
}

}  // namespace

int main() {
  verify_case_suite({1, 2, 3, 4, 5}, 3);
  verify_case_suite({7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}, 3);
  verify_case_suite({cvo::KDTREE_K_SIZE - 1, cvo::KDTREE_K_SIZE, cvo::KDTREE_K_SIZE + 1}, 3);
  verify_k_boundaries();
  verify_special_clouds();
  verify_permutation_invariance(33, 5);

  std::cout << "cuKdTree tests passed." << std::endl;
  return 0;
}
