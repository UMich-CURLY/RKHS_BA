#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#include "cvo/const_def.hpp"

namespace perl_registration {

template <typename T>
struct KernelArray {
  T* data;
  int size;

  KernelArray(thrust::device_vector<T>& dvec)
      : data(thrust::raw_pointer_cast(dvec.data())), size(static_cast<int>(dvec.size())) {}

  __host__ __device__ KernelArray(T* data_, int size_) : data(data_), size(size_) {}
};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

#define cudaSafe(ans) \
  { gpuAssert((ans), __FILE__, __LINE__, true); }

inline constexpr int KDTREE_K_SIZE = cvo::KDTREE_K_SIZE;

struct TreeNode {
  int point = -1;
  int axis = -1;
  int leftChild = -1;
  int rightChild = -1;
};

struct Range_t {
  int start = -1;
  int stop = -1;
};

struct Results_t {
  int parrent_id = -1;
  Range_t range;
  bool left = false;
};

struct isActive {
  __host__ __device__ bool operator()(const Results_t& r) const {
    return r.range.start > -1;
  }
};

template <typename PointT>
struct ZipPointAxisLess {
  int axis;

  __host__ __device__ explicit ZipPointAxisLess(int axis_) : axis(axis_) {}

  __host__ __device__ bool operator()(const thrust::tuple<PointT, int>& lhs,
                                      const thrust::tuple<PointT, int>& rhs) const {
    return thrust::get<0>(lhs).data[axis] < thrust::get<0>(rhs).data[axis];
  }
};

class cuPQueue {
 private:
  float max_ = -1.0f;
  int size_ = -1;
  int top_ = -1;
  float priorities_[KDTREE_K_SIZE];
  int* values_ = nullptr;

 public:
  __host__ __device__ cuPQueue() = default;

  __host__ __device__ explicit cuPQueue(const int& size) : max_(2e16f), size_(size), top_(-1) {}

  __host__ __device__ void push(const float& p, const int& v) {
    if (size_ <= 0) {
      return;
    }
    if (top_ >= 0 && top_ + 1 == size_ && p >= priorities_[top_]) {
      return;
    }

    const int filled = top_ + 1;
    const bool has_room = filled < size_;
    if (has_room) {
      ++top_;
    }

    int pos = has_room ? top_ : size_ - 1;
    while (pos > 0 && priorities_[pos - 1] > p) {
      priorities_[pos] = priorities_[pos - 1];
      values_[pos] = values_[pos - 1];
      --pos;
    }
    priorities_[pos] = p;
    values_[pos] = v;

    if (top_ + 1 == size_) {
      max_ = priorities_[top_];
    } else {
      max_ = 2e16f;
    }
  }

  __host__ __device__ void set_value_ptr(int* val) { values_ = val; }
  __host__ __device__ float get_max() const { return max_; }
};

template <typename PointT>
__host__ __device__ __forceinline__ int biggestAxis(const PointT& p) {
  if (p.data[0] > p.data[1] && p.data[0] > p.data[2]) return 0;
  if (p.data[1] > p.data[2]) return 1;
  return 2;
}

template <typename PointT>
__host__ __device__ PointT PointDifference(const PointT& a, const PointT& b) {
  return PointT(a.x - b.x, a.y - b.y, a.z - b.z);
}

template <typename PointT>
__host__ __device__ PointT PointMax(const PointT& a, const PointT& b) {
  return PointT(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

template <typename PointT>
__host__ __device__ PointT PointMin(const PointT& a, const PointT& b) {
  return PointT(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

template <typename PointT>
struct MinMaxVec {
  PointT min;
  PointT max;
};

template <typename PointT>
struct MinMaxUnary {
  __host__ __device__ MinMaxVec<PointT> operator()(const PointT& in) const {
    return {in, in};
  }
};

template <typename PointT>
struct MinMaxBinary {
  __host__ __device__ MinMaxVec<PointT> operator()(const MinMaxVec<PointT>& a,
                                                   const MinMaxVec<PointT>& b) const {
    return {PointMin<PointT>(a.min, b.min), PointMax<PointT>(a.max, b.max)};
  }
};

template <typename PointT>
struct sortByAxis {
  int axis;
  __host__ __device__ explicit sortByAxis(const int& a) : axis(a) {}
  __host__ __device__ bool operator()(const PointT& a, const PointT& b) const {
    return a.data[axis] < b.data[axis];
  }
};

template <typename PointT>
__device__ __forceinline__ int computeVar(KernelArray<PointT> d_points, const int& start, const int& stop) {
  float sum[3] = {0.0f, 0.0f, 0.0f};
  float sqSum[3] = {0.0f, 0.0f, 0.0f};
  float var[3] = {0.0f, 0.0f, 0.0f};
  for (int i = start; i <= stop; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum[j] += d_points.data[i].data[j];
      sqSum[j] += d_points.data[i].data[j] * d_points.data[i].data[j];
    }
  }
  int n = stop + 1 - start;
  int maxIdx = -1;
  float maxVar = -1.0f;
  for (int j = 0; j < 3; ++j) {
    var[j] = (sqSum[j] - (sum[j] * sum[j]) / n) / (n - 1);
    if (var[j] > maxVar) {
      maxVar = var[j];
      maxIdx = j;
    }
  }
  return maxIdx;
}

template <typename PointT>
__device__ __forceinline__ int partition(KernelArray<PointT> d_points,
                                         KernelArray<int> d_indices,
                                         const int& start,
                                         const int& stop,
                                         const int& pivot,
                                         const int& axis) {
  const float pivotVal = d_points.data[pivot].data[axis];
  int storeIdx = 0;
  for (int i = start; i <= stop; ++i) {
    if (d_points.data[i].data[axis] < pivotVal) {
      thrust::swap(d_points.data[storeIdx + start], d_points.data[i]);
      thrust::swap(d_indices.data[storeIdx + start], d_indices.data[i]);
      ++storeIdx;
    }
  }
  return start + storeIdx;
}

template <typename PointT>
__device__ __forceinline__ void qselect(KernelArray<PointT> d_points,
                                        KernelArray<int> d_indices,
                                        int start,
                                        int stop,
                                        int n,
                                        const int& axis) {
  curandState_t state;
  curand_init(0, 0, 0, &state);
  while (true) {
    if (start == stop) return;
    const int max = stop - start;
    int pivot = curand(&state) % max;
    thrust::swap(d_points.data[pivot + start], d_points.data[stop]);
    thrust::swap(d_indices.data[pivot + start], d_indices.data[stop]);
    pivot = partition<PointT>(d_points, d_indices, start, stop - 1, stop, axis);
    thrust::swap(d_points.data[pivot], d_points.data[stop]);
    thrust::swap(d_indices.data[pivot], d_indices.data[stop]);
    if (n == pivot) {
      return;
    } else if (n < pivot) {
      stop = pivot - 1;
    } else {
      start = pivot + 1;
    }
  }
}

template <typename PointT>
__global__ void smallNodeStage(KernelArray<PointT> d_points,
                               KernelArray<int> d_indices,
                               KernelArray<Results_t> d_results,
                               KernelArray<Results_t> d_results_out,
                               KernelArray<TreeNode> d_tree,
                               int current_index,
                               int n) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  const int id = current_index + pos;

  if (d_results.data[pos].left) {
    d_tree.data[d_results.data[pos].parrent_id].leftChild = id;
  } else {
    d_tree.data[d_results.data[pos].parrent_id].rightChild = id;
  }

  const int start = d_results.data[pos].range.start;
  const int stop = d_results.data[pos].range.stop;
  const int axis = computeVar<PointT>(d_points, start, stop);
  const int median = static_cast<int>(((stop + 1.0 - start) / 2.0) + start);

  qselect<PointT>(d_points, d_indices, start, stop, median, axis);

  d_tree.data[id].point = median;
  d_tree.data[id].axis = axis;

  d_results_out.data[2 * pos].parrent_id = id;
  d_results_out.data[2 * pos].left = true;
  if (median > start) {
    d_results_out.data[2 * pos].range.start = start;
    d_results_out.data[2 * pos].range.stop = median - 1;
  } else {
    d_results_out.data[2 * pos].range.start = -1;
    d_results_out.data[2 * pos].range.stop = -1;
  }

  d_results_out.data[2 * pos + 1].parrent_id = id;
  d_results_out.data[2 * pos + 1].left = false;
  if (median < stop) {
    d_results_out.data[2 * pos + 1].range.start = median + 1;
    d_results_out.data[2 * pos + 1].range.stop = stop;
  } else {
    d_results_out.data[2 * pos + 1].range.start = -1;
    d_results_out.data[2 * pos + 1].range.stop = -1;
  }
}

template <typename PointT>
void largeNodeStage(thrust::device_vector<PointT>& d_points,
                    thrust::device_vector<int>& d_indices,
                    thrust::host_vector<Results_t>& results,
                    thrust::host_vector<Results_t>& results_new,
                    thrust::host_vector<TreeNode>& tree,
                    int p,
                    int level) {
  thrust::host_vector<PointT> host_points = d_points;
#pragma omp parallel for
  for (int i = 0; i < p; ++i) {
    const int treeIdx = static_cast<int>(std::pow(2, level)) + i - 1;
    const int start = results[i].range.start;
    const int stop = results[i].range.stop;
    const int median = static_cast<int>(((stop + 1 - start) / 2.0) + start);

    PointT min_pt(FLT_MAX, FLT_MAX, FLT_MAX);
    PointT max_pt(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int idx = start; idx <= stop; ++idx) {
      min_pt = PointMin<PointT>(min_pt, host_points[idx]);
      max_pt = PointMax<PointT>(max_pt, host_points[idx]);
    }

    const int axis = biggestAxis<PointT>(PointDifference(max_pt, min_pt));
    const int idLeft = static_cast<int>(std::pow(2, level + 1)) + i * 2 - 1;
    const int idRight = static_cast<int>(std::pow(2, level + 1)) + i * 2;
    const bool has_left = median > start;
    const bool has_right = median < stop;
    tree[treeIdx].point = median;
    tree[treeIdx].axis = axis;
    tree[treeIdx].leftChild = has_left ? idLeft : -1;
    tree[treeIdx].rightChild = has_right ? idRight : -1;

    results_new[2 * i].parrent_id = treeIdx;
    results_new[2 * i].left = true;
    results_new[2 * i].range.start = (median > start) ? start : -1;
    results_new[2 * i].range.stop = (median > start) ? median - 1 : -1;

    results_new[2 * i + 1].parrent_id = treeIdx;
    results_new[2 * i + 1].left = false;
    results_new[2 * i + 1].range.start = (median < stop) ? median + 1 : -1;
    results_new[2 * i + 1].range.stop = (median < stop) ? stop : -1;

    auto points_begin = d_points.begin() + start;
    auto points_end = d_points.begin() + stop + 1;
    auto indices_begin = d_indices.begin() + start;
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(points_begin, indices_begin));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(points_end, indices_begin + (stop + 1 - start)));
    thrust::sort(
        thrust::device,
        zip_begin,
        zip_end,
        ZipPointAxisLess<PointT>(axis));
  }
}

template <typename PointT>
__device__ __inline__ float pDist(const PointT& a, const PointT& b) {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}

template <typename PointT>
__device__ void TreeSearch(KernelArray<PointT> d_points,
                           KernelArray<int> d_original_indices,
                           KernelArray<TreeNode> d_tree,
                           const PointT& query_point,
                           KernelArray<int> nearest_indices,
                           int k) {
  cuPQueue pq(k);
  pq.set_value_ptr(nearest_indices.data);
  float max = pq.get_max();

  int visit_stack[50];
  int visit_pos = 0;
  int back_stack[50];
  int back_pos = 0;
  visit_stack[visit_pos++] = 0;

  while (visit_pos > 0 || back_pos > 0) {
    while (visit_pos > 0) {
      const int current_node = visit_stack[--visit_pos];
      if (current_node < 0) break;

      const int& node_point = d_tree.data[current_node].point;
      const int& axis = d_tree.data[current_node].axis;
      const int& left_node = d_tree.data[current_node].leftChild;
      const int& right_node = d_tree.data[current_node].rightChild;
      const float current_dist = pDist<PointT>(d_points.data[node_point], query_point);

      if (current_dist < max) {
        pq.push(current_dist, d_original_indices.data[node_point]);
        max = pq.get_max();
      }

      if (axis < 0) break;

      if (query_point.data[axis] < d_points.data[node_point].data[axis]) {
        visit_stack[visit_pos++] = left_node;
        back_stack[back_pos++] = current_node;
      } else {
        visit_stack[visit_pos++] = right_node;
        back_stack[back_pos++] = current_node;
      }
    }

    if (back_pos > 0) {
      const int current_node = back_stack[--back_pos];
      if (current_node >= 0) {
        const int& node_point = d_tree.data[current_node].point;
        const int& axis = d_tree.data[current_node].axis;
        const int& left_node = d_tree.data[current_node].leftChild;
        const int& right_node = d_tree.data[current_node].rightChild;
        float axis_dist = query_point.data[axis] - d_points.data[node_point].data[axis];
        axis_dist *= axis_dist;

        if (axis_dist < max) {
          if (query_point.data[axis] < d_points.data[node_point].data[axis]) {
            visit_stack[visit_pos++] = right_node;
          } else {
            visit_stack[visit_pos++] = left_node;
          }
        }
      }
    }
  }
}

template <typename PointT>
__global__ void NKSearch(KernelArray<PointT> d_points,
                         KernelArray<int> d_original_indices,
                         KernelArray<TreeNode> d_tree,
                         KernelArray<PointT> query_points,
                         KernelArray<int> nearest_indices,
                         int k) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= query_points.size) return;

  KernelArray<int> indices(nearest_indices.data + pos * k, k);
  TreeSearch<PointT>(d_points, d_original_indices, d_tree, query_points.data[pos], indices, k);
}

template <typename PointT>
class cuKdTree {
 private:
  thrust::device_vector<PointT>* d_point_cloud_ = nullptr;
  thrust::device_vector<int> d_original_indices_;
  thrust::device_vector<TreeNode> d_tree_;
  bool profile_ = false;
  bool input_cloud_set_ = false;
  float epsilon_ = std::numeric_limits<float>::epsilon();
  size_t n_tree_points_ = 0;

  static constexpr int stage_pivot = 13;
  static constexpr int thread_num = 512;

 public:
  using PointType = PointT;
  using SharedPtr = std::shared_ptr<cuKdTree<PointT>>;
  using SharedConstPtr = std::shared_ptr<const cuKdTree<PointT>>;

  void SetEpsilon(const float epsilon) { epsilon_ = epsilon; }
  bool IsInputCloudSet() const { return input_cloud_set_; }
  const thrust::device_vector<TreeNode>& tree() const { return d_tree_; }

  void SetInputCloud(thrust::device_vector<PointT>& d_cloud) {
    n_tree_points_ = d_cloud.size();
    if (n_tree_points_ == 0) {
      d_tree_.clear();
      d_point_cloud_ = &d_cloud;
      input_cloud_set_ = true;
      return;
    }

    const int log_tree_points = static_cast<int>(std::floor(std::log2(n_tree_points_)));
    const int log2n = std::max(0, log_tree_points - stage_pivot);
    const int large_n = static_cast<int>(std::pow(2, log2n));
    const int frontier_capacity = std::max(static_cast<int>(n_tree_points_), 2 * large_n);
    d_point_cloud_ = &d_cloud;
    input_cloud_set_ = true;

    d_tree_.resize(n_tree_points_);
    d_original_indices_.resize(n_tree_points_);
    thrust::sequence(thrust::device, d_original_indices_.begin(), d_original_indices_.end(), 0);
    thrust::host_vector<TreeNode> host_tree(n_tree_points_);
    TreeNode tempNode;
    thrust::fill(thrust::device, d_tree_.begin(), d_tree_.end(), tempNode);
    thrust::fill(thrust::host, host_tree.begin(), host_tree.end(), tempNode);

    Results_t temp_result;
    temp_result.range.start = 0;
    temp_result.range.stop = static_cast<int>(n_tree_points_) - 1;

    thrust::device_vector<Results_t> d_results(frontier_capacity), d_new_results(frontier_capacity);
    thrust::host_vector<Results_t> host_results(frontier_capacity), host_new_results(frontier_capacity);
    thrust::fill_n(thrust::host, host_results.begin(), 1, temp_result);

    for (int i = 0; i <= log2n; ++i) {
      const int p = static_cast<int>(std::pow(2, i));
      largeNodeStage<PointT>(d_cloud, d_original_indices_, host_results, host_new_results, host_tree, p, i);
      cudaSafe(cudaDeviceSynchronize());
      thrust::copy(thrust::host, host_new_results.begin(), host_new_results.begin() + 2 * p, host_results.begin());
    }

    thrust::copy(host_new_results.begin(), host_new_results.end(), d_new_results.begin());
    thrust::copy(host_results.begin(), host_results.end(), d_results.begin());
    thrust::copy(host_tree.begin(), host_tree.end(), d_tree_.begin());

    int current_index = large_n * 2 - 1;
    auto last_active = thrust::copy_if(thrust::device,
                                       d_new_results.begin(),
                                       d_new_results.begin() + 2 * large_n,
                                       d_results.begin(),
                                       isActive());
    int n_active = static_cast<int>(thrust::distance(d_results.begin(), last_active));

    while (n_active > 0) {
      const int blocks = static_cast<int>(std::ceil(static_cast<float>(n_active) / static_cast<float>(thread_num)));
      smallNodeStage<PointT><<<blocks, thread_num>>>(
          d_cloud,
          d_original_indices_,
          d_results,
          d_new_results,
          d_tree_,
          current_index,
          n_active);
      current_index += n_active;
      cudaSafe(cudaDeviceSynchronize());
      last_active = thrust::copy_if(thrust::device, d_new_results.begin(), d_new_results.begin() + 2 * n_active,
                                    d_results.begin(), isActive());
      n_active = thrust::distance(d_results.begin(), last_active);
    }
  }

  int NearestKSearch(const thrust::device_vector<PointT>& d_query_points,
                     int k,
                     thrust::device_vector<int>& indices) {
    indices.resize(d_query_points.size() * k);
    const int blocks = static_cast<int>(std::ceil(static_cast<float>(d_query_points.size()) / static_cast<float>(thread_num)));
    auto& query_nonconst = const_cast<thrust::device_vector<PointT>&>(d_query_points);
    NKSearch<PointT><<<blocks, thread_num>>>(
        *d_point_cloud_,
        d_original_indices_,
        d_tree_,
        query_nonconst,
        indices,
        k);
    cudaSafe(cudaDeviceSynchronize());
    return k;
  }
};

}  // namespace perl_registration
