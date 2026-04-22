#pragma once

#include <type_traits>

#include <pcl/point_cloud.h>

namespace cvo {
namespace point_converter {

template <typename...>
using void_t = void;

template <typename T, typename = void>
struct has_x : std::false_type {};
template <typename T>
struct has_x<T, void_t<decltype(std::declval<T&>().x)>> : std::true_type {};

template <typename T, typename = void>
struct has_y : std::false_type {};
template <typename T>
struct has_y<T, void_t<decltype(std::declval<T&>().y)>> : std::true_type {};

template <typename T, typename = void>
struct has_z : std::false_type {};
template <typename T>
struct has_z<T, void_t<decltype(std::declval<T&>().z)>> : std::true_type {};

template <typename T, typename = void>
struct has_r : std::false_type {};
template <typename T>
struct has_r<T, void_t<decltype(std::declval<T&>().r)>> : std::true_type {};

template <typename T, typename = void>
struct has_g : std::false_type {};
template <typename T>
struct has_g<T, void_t<decltype(std::declval<T&>().g)>> : std::true_type {};

template <typename T, typename = void>
struct has_b : std::false_type {};
template <typename T>
struct has_b<T, void_t<decltype(std::declval<T&>().b)>> : std::true_type {};

template <typename T, typename = void>
struct has_intensity : std::false_type {};
template <typename T>
struct has_intensity<T, void_t<decltype(std::declval<T&>().intensity)>> : std::true_type {};

template <typename T, typename = void>
struct has_label : std::false_type {};
template <typename T>
struct has_label<T, void_t<decltype(std::declval<T&>().label)>> : std::true_type {};

template <typename T, typename = void>
struct has_features_array : std::false_type {};
template <typename T>
struct has_features_array<T, void_t<decltype(std::declval<T&>().features[0])>> : std::true_type {};

template <typename T, typename = void>
struct has_normal_array : std::false_type {};
template <typename T>
struct has_normal_array<T, void_t<decltype(std::declval<T&>().normal[0])>> : std::true_type {};

template <typename T, typename = void>
struct has_normal_x : std::false_type {};
template <typename T>
struct has_normal_x<T, void_t<decltype(std::declval<T&>().normal_x)>> : std::true_type {};

template <typename T, typename = void>
struct has_normal_y : std::false_type {};
template <typename T>
struct has_normal_y<T, void_t<decltype(std::declval<T&>().normal_y)>> : std::true_type {};

template <typename T, typename = void>
struct has_normal_z : std::false_type {};
template <typename T>
struct has_normal_z<T, void_t<decltype(std::declval<T&>().normal_z)>> : std::true_type {};

template <typename DstPoint, typename SrcPoint>
void convert_point(const SrcPoint& src, DstPoint& dst) {
    dst = DstPoint{};

    if constexpr (has_x<DstPoint>::value && has_x<SrcPoint>::value) dst.x = src.x;
    if constexpr (has_y<DstPoint>::value && has_y<SrcPoint>::value) dst.y = src.y;
    if constexpr (has_z<DstPoint>::value && has_z<SrcPoint>::value) dst.z = src.z;

    if constexpr (has_r<DstPoint>::value && has_r<SrcPoint>::value) dst.r = src.r;
    if constexpr (has_g<DstPoint>::value && has_g<SrcPoint>::value) dst.g = src.g;
    if constexpr (has_b<DstPoint>::value && has_b<SrcPoint>::value) dst.b = src.b;

    if constexpr (has_intensity<DstPoint>::value && has_intensity<SrcPoint>::value) {
        dst.intensity = src.intensity;
    }

    if constexpr (has_label<DstPoint>::value && has_label<SrcPoint>::value) {
        dst.label = src.label;
    }

    if constexpr (has_features_array<DstPoint>::value) {
        if constexpr (has_features_array<SrcPoint>::value) {
            constexpr std::size_t dst_dim = sizeof(dst.features) / sizeof(dst.features[0]);
            constexpr std::size_t src_dim = sizeof(src.features) / sizeof(src.features[0]);
            const std::size_t copy_dim = dst_dim < src_dim ? dst_dim : src_dim;
            for (std::size_t i = 0; i < copy_dim; ++i) {
                dst.features[i] = src.features[i];
            }
        } else if constexpr (has_r<SrcPoint>::value && has_g<SrcPoint>::value && has_b<SrcPoint>::value) {
            constexpr std::size_t dst_dim = sizeof(dst.features) / sizeof(dst.features[0]);
            if constexpr (dst_dim > 0) dst.features[0] = static_cast<float>(src.r) / 255.0f;
            if constexpr (dst_dim > 1) dst.features[1] = static_cast<float>(src.g) / 255.0f;
            if constexpr (dst_dim > 2) dst.features[2] = static_cast<float>(src.b) / 255.0f;
        } else if constexpr (has_intensity<SrcPoint>::value) {
            dst.features[0] = static_cast<float>(src.intensity)  / 255.0f;
        }
    }

    if constexpr (has_normal_array<DstPoint>::value) {
        if constexpr (has_normal_array<SrcPoint>::value) {
            dst.normal[0] = src.normal[0];
            dst.normal[1] = src.normal[1];
            dst.normal[2] = src.normal[2];
        } else if constexpr (has_normal_x<SrcPoint>::value && has_normal_y<SrcPoint>::value && has_normal_z<SrcPoint>::value) {
            dst.normal[0] = src.normal_x;
            dst.normal[1] = src.normal_y;
            dst.normal[2] = src.normal_z;
        }
    }

    if constexpr (has_normal_x<DstPoint>::value && has_normal_y<DstPoint>::value && has_normal_z<DstPoint>::value) {
        if constexpr (has_normal_array<SrcPoint>::value) {
            dst.normal_x = src.normal[0];
            dst.normal_y = src.normal[1];
            dst.normal_z = src.normal[2];
        } else if constexpr (has_normal_x<SrcPoint>::value && has_normal_y<SrcPoint>::value && has_normal_z<SrcPoint>::value) {
            dst.normal_x = src.normal_x;
            dst.normal_y = src.normal_y;
            dst.normal_z = src.normal_z;
        }
    }
}

template <typename DstPoint, typename SrcPoint>
void convert_point_cloud(const pcl::PointCloud<SrcPoint>& src,
                         pcl::PointCloud<DstPoint>& dst) {
    dst.clear();
    dst.reserve(src.size());
    for (const auto& point : src) {
        DstPoint converted;
        convert_point(point, converted);
        dst.push_back(converted);
    }
    dst.header = src.header;
    dst.width = static_cast<std::uint32_t>(dst.size());
    dst.height = 1;
    dst.is_dense = src.is_dense;
}

} // namespace point_converter
} // namespace cvo
