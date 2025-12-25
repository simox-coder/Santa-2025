/**
 * Santa 2025 Solver - C++ Collision Detection
 * 
 * High-performance collision detection using pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <array>
#include <algorithm>

namespace py = pybind11;

// Project polygon onto axis and return (min, max)
std::pair<double, double> project_polygon(
    const py::array_t<double>& vertices,
    double axis_x, double axis_y
) {
    auto v = vertices.unchecked<2>();
    int n = v.shape(0);
    
    double proj = v(0, 0) * axis_x + v(0, 1) * axis_y;
    double min_proj = proj;
    double max_proj = proj;
    
    for (int i = 1; i < n; ++i) {
        proj = v(i, 0) * axis_x + v(i, 1) * axis_y;
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }
    
    return {min_proj, max_proj};
}

// Check if two convex polygons overlap using SAT
bool check_convex_overlap(
    const py::array_t<double>& vertices1,
    const py::array_t<double>& vertices2
) {
    auto v1 = vertices1.unchecked<2>();
    auto v2 = vertices2.unchecked<2>();
    
    int n1 = v1.shape(0);
    int n2 = v2.shape(0);
    
    // Check axes from polygon 1
    for (int i = 0; i < n1; ++i) {
        int next_i = (i + 1) % n1;
        double edge_x = v1(next_i, 0) - v1(i, 0);
        double edge_y = v1(next_i, 1) - v1(i, 1);
        
        // Perpendicular axis
        double axis_x = -edge_y;
        double axis_y = edge_x;
        
        // Normalize
        double length = std::sqrt(axis_x * axis_x + axis_y * axis_y);
        if (length > 1e-10) {
            axis_x /= length;
            axis_y /= length;
        }
        
        auto [min1, max1] = project_polygon(vertices1, axis_x, axis_y);
        auto [min2, max2] = project_polygon(vertices2, axis_x, axis_y);
        
        if (max1 <= min2 || max2 <= min1) {
            return false;  // Separation found
        }
    }
    
    // Check axes from polygon 2
    for (int i = 0; i < n2; ++i) {
        int next_i = (i + 1) % n2;
        double edge_x = v2(next_i, 0) - v2(i, 0);
        double edge_y = v2(next_i, 1) - v2(i, 1);
        
        double axis_x = -edge_y;
        double axis_y = edge_x;
        
        double length = std::sqrt(axis_x * axis_x + axis_y * axis_y);
        if (length > 1e-10) {
            axis_x /= length;
            axis_y /= length;
        }
        
        auto [min1, max1] = project_polygon(vertices1, axis_x, axis_y);
        auto [min2, max2] = project_polygon(vertices2, axis_x, axis_y);
        
        if (max1 <= min2 || max2 <= min1) {
            return false;
        }
    }
    
    return true;  // No separation found
}

// Extract triangle (foliage) from tree vertices
py::array_t<double> extract_triangle(const py::array_t<double>& tree_vertices) {
    auto v = tree_vertices.unchecked<2>();
    
    py::array_t<double> triangle({3, 2});
    auto t = triangle.mutable_unchecked<2>();
    
    // Triangle: tip (0), left foliage (1), right foliage (6)
    t(0, 0) = v(0, 0); t(0, 1) = v(0, 1);
    t(1, 0) = v(1, 0); t(1, 1) = v(1, 1);
    t(2, 0) = v(6, 0); t(2, 1) = v(6, 1);
    
    return triangle;
}

// Extract trunk rectangle from tree vertices
py::array_t<double> extract_trunk(const py::array_t<double>& tree_vertices) {
    auto v = tree_vertices.unchecked<2>();
    
    py::array_t<double> trunk({4, 2});
    auto t = trunk.mutable_unchecked<2>();
    
    // Trunk: indices 2, 3, 4, 5
    t(0, 0) = v(2, 0); t(0, 1) = v(2, 1);
    t(1, 0) = v(3, 0); t(1, 1) = v(3, 1);
    t(2, 0) = v(4, 0); t(2, 1) = v(4, 1);
    t(3, 0) = v(5, 0); t(3, 1) = v(5, 1);
    
    return trunk;
}

// Check if two tree polygons overlap
bool check_tree_overlap(
    const py::array_t<double>& vertices1,
    const py::array_t<double>& vertices2
) {
    // Decompose into convex parts
    auto tri1 = extract_triangle(vertices1);
    auto trunk1 = extract_trunk(vertices1);
    auto tri2 = extract_triangle(vertices2);
    auto trunk2 = extract_trunk(vertices2);
    
    // Check all 4 pairs
    if (check_convex_overlap(tri1, tri2)) return true;
    if (check_convex_overlap(tri1, trunk2)) return true;
    if (check_convex_overlap(trunk1, tri2)) return true;
    if (check_convex_overlap(trunk1, trunk2)) return true;
    
    return false;
}

PYBIND11_MODULE(santa2025_collision_cpp, m) {
    m.doc() = "Santa 2025 - High-performance collision detection";
    
    m.def("check_tree_overlap", &check_tree_overlap,
          "Check if two tree polygons overlap",
          py::arg("vertices1"), py::arg("vertices2"));
    
    m.def("check_convex_overlap", &check_convex_overlap,
          "Check if two convex polygons overlap",
          py::arg("vertices1"), py::arg("vertices2"));
}
