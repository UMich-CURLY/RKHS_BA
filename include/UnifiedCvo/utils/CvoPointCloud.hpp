#pragma once

#include <vector>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>

#include "utils/PointConverter.hpp"
#include "utils/PointSemantic.hpp"

namespace cvo {

  template <typename PointT = pcl::PointSemantic<3, 19>>
  class CvoPointCloud {
  public:
    using PointType = PointT;
    using Container = std::vector<PointT>;

    // -----------------------------------------------------------------------
    // Constructors / Destructor
    // -----------------------------------------------------------------------
    CvoPointCloud() = default;
    explicit CvoPointCloud(size_t reserve_size) {
      points_.reserve(reserve_size);
    }
    template <typename OtherPointT>
    explicit CvoPointCloud(const pcl::PointCloud<OtherPointT>& cloud) {
      points_.reserve(cloud.size());
      for (const auto& point : cloud) {
        PointT converted;
        point_converter::convert_point(point, converted);
        points_.push_back(converted);
      }
    }
    ~CvoPointCloud() = default;

    // Copy / move constructors and assignment
    CvoPointCloud(const CvoPointCloud&) = default;
    CvoPointCloud(CvoPointCloud&&) noexcept = default;
    CvoPointCloud& operator=(const CvoPointCloud&) = default;
    CvoPointCloud& operator=(CvoPointCloud&&) noexcept = default;

    // -----------------------------------------------------------------------
    // Container-like operations
    // -----------------------------------------------------------------------
    void push_back(const PointT& pt) {
      points_.push_back(pt);
    }
    void push_back(PointT&& pt) {
      points_.push_back(std::move(pt));
    }
    void pop_back() {
      points_.pop_back();
    }
    void resize(size_t count) {
      points_.resize(count);
    }
    void resize(size_t count, const PointT& value) {
      points_.resize(count, value);
    }
    void clear() {
      points_.clear();
    }
    void reserve(size_t capacity) {
      points_.reserve(capacity);
    }

    size_t size() const noexcept {
      return points_.size();
    }
    bool empty() const noexcept {
      return points_.empty();
    }

    // Element access
    PointT& at(size_t index) {
      return points_.at(index);
    }
    const PointT& at(size_t index) const {
      return points_.at(index);
    }
    PointT& operator[](size_t index) {
      return points_[index];
    }
    const PointT& operator[](size_t index) const {
      return points_[index];
    }

    // Direct access to underlying vector
    Container& points() { return points_; }
    const Container& points() const { return points_; }

    // -----------------------------------------------------------------------
    // Convenience methods for typical point attributes (if available)
    // These rely on the point type having the corresponding members.
    // They can be enabled via SFINAE or concepts (omitted for brevity).
    // -----------------------------------------------------------------------
    float x_at(size_t i) const { return points_[i].x; }
    float y_at(size_t i) const { return points_[i].y; }
    float z_at(size_t i) const { return points_[i].z; }
    Eigen::Vector3f position_at(size_t i) const {
      return Eigen::Vector3f(points_[i].x, points_[i].y, points_[i].z);
    }

    // -----------------------------------------------------------------------
    // Static transformation helper (works for any PointT that has x,y,z)
    // -----------------------------------------------------------------------
    static void transform(const Eigen::Matrix4f& pose,
                          const CvoPointCloud<PointT>& input,
                          CvoPointCloud<PointT>& output) {
      output.resize(input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        const PointT& src = input[i];
        PointT& dst = output[i];
        Eigen::Vector4f p(src.x, src.y, src.z, 1.0f);
        Eigen::Vector4f tp = pose * p;
        dst = src;  // copy all other fields
        dst.x = tp.x();
        dst.y = tp.y();
        dst.z = tp.z();
      }
    }

  private:
    Container points_;
  };

  // -----------------------------------------------------------------------
  // Non‑member functions (for convenience)
  // -----------------------------------------------------------------------
  template <typename PointT>
  CvoPointCloud<PointT> operator+(CvoPointCloud<PointT> a, const CvoPointCloud<PointT>& b) {
    a.points().insert(a.points().end(), b.points().begin(), b.points().end());
    return a;
  }

  // For backward compatibility, keep the old name as an alias.
  using point_cloud = CvoPointCloud<>;

} // namespace cvo
