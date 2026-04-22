#include "cvo/LieGroup.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <Eigen/Geometry>

using namespace cvo;

const double EPS_D = 1e-10;
const float  EPS_F = 1e-5f;

// Helper to compare Eigen expressions of any fixed/dynamic size.
template <typename DerivedA, typename DerivedB>
bool isApprox(const Eigen::MatrixBase<DerivedA>& a,
              const Eigen::MatrixBase<DerivedB>& b,
              typename DerivedA::Scalar eps) {
    return (a.derived() - b.derived()).norm() <= eps;
}

template<typename T>
void test_skew_unskew() {
    Eigen::Matrix<T, 3, 1> v(T(1), T(2), T(3));
    auto S = skew<T>(v);
    auto u = unskew<T>(S);
    assert(isApprox(u, v, T(1e-6)));

    // Check skew symmetry
    assert(std::abs(S(0,0)) < T(1e-6));
    assert(std::abs(S(1,1)) < T(1e-6));
    assert(std::abs(S(2,2)) < T(1e-6));
    assert(std::abs(S(0,1) + S(1,0)) < T(1e-6));
    assert(std::abs(S(0,2) + S(2,0)) < T(1e-6));
    assert(std::abs(S(1,2) + S(2,1)) < T(1e-6));

    // Check row-major version
    auto S_row = skew<T, Eigen::RowMajor>(v);
    assert(isApprox(S_row.template cast<T>(), S.template cast<T>(), T(1e-6)));
    std::cout << "[PASS] skew/unskew<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_hat2_wedge() {
    Eigen::Matrix<T, 6, 1> xi;
    xi << T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6);
    Eigen::Matrix<T, 4, 4> X = hat2<T>(xi);
    Eigen::Matrix<T, 6, 1> xi2 = wedge<T>(X);
    assert(isApprox(xi, xi2, T(1e-6)));
    std::cout << "[PASS] hat2/wedge<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_SO3_Jacobians(T eps) {
    Eigen::Matrix<T, 3, 1> w(T(0.1), T(0.2), T(0.3));
    // Left Jacobian near zero
    Eigen::Matrix<T, 3, 1> w0 = Eigen::Matrix<T, 3, 1>::Zero();
    auto J0 = LeftJacobian_SO3<T>(w0);
    assert(isApprox(J0, Eigen::Matrix<T, 3, 3>::Identity(), eps));

    // Left Jacobian inverse property: Jl * Jl^{-1} = I
    auto Jl = LeftJacobian_SO3<T>(w);
    auto Jl_inv = LeftJacobianInverse_SO3<T>(w);
    assert(isApprox(Jl * Jl_inv, Eigen::Matrix<T, 3, 3>::Identity(), eps));

    // Jl * w = w (should hold exactly for SO(3) left Jacobian)
    Eigen::Matrix<T, 3, 1> Jl_w = Jl * w;
    assert(isApprox(Jl_w, w, eps * T(10)));

    std::cout << "[PASS] SO3 Jacobians<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_SE3_Jacobians(T eps) {
    Eigen::Matrix<T, 6, 1> xi;
    xi << T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6);
    auto Jl = LeftJacobian_SE3<T>(xi);
    auto Jr = RightJacobian_SE3<T>(xi);
    auto Jr_inv = RightJacobianInverse_SE3<T>(xi);

    // Jr * Jr_inv = I
    assert(isApprox(Jr * Jr_inv, Eigen::Matrix<T, 6, 6>::Identity(), eps));

    // Convention used here: Jr(xi) = Jl(-xi)
    auto Jl_neg = LeftJacobian_SE3<T>(-xi);
    assert(isApprox(Jl_neg, Jr, eps * T(10)));

    std::cout << "[PASS] SE3 Jacobians<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_SO3_exp_log(T eps) {
    Eigen::Matrix<T, 3, 1> w(T(0.1), T(-0.2), T(0.05));
    Eigen::Matrix<T, 3, 3> R = Exp_SO3<T>(w);
    // R should be orthogonal
    assert(isApprox(R * R.transpose(), Eigen::Matrix<T, 3, 3>::Identity(), eps));
    // log(exp(w)) = w
    Eigen::Matrix<T, 3, 1> w2 = Log_SO3<T>(R);
    // The logarithm may return an equivalent rotation vector (same direction but possibly different magnitude mod 2pi)
    // For small angles it should be identical.
    assert(isApprox(w, w2, eps));

    // Test row-major version
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R_row = Exp_SO3<T, Eigen::RowMajor>(w);
    assert(isApprox(R_row, R, eps));

    std::cout << "[PASS] SO3 exp/log<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_SE3_exp_log(T eps) {
    Eigen::Matrix<T, 6, 1> xi;
    xi << T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6);
    Eigen::Matrix<T, 4, 4> Tmat = Exp_SE3<T>(xi);
    Eigen::Matrix<T, 6, 1> xi2 = Log_SE3<T>(Tmat);
    assert(isApprox(xi, xi2, eps));

    // Test 3x4 version
    Eigen::Matrix<T, 3, 4> T34 = Exp_SE3<T>(xi, true);
    assert(isApprox(T34.template block<3,3>(0,0), Tmat.template block<3,3>(0,0), eps));
    assert(isApprox(T34.template block<3,1>(0,3), Tmat.template block<3,1>(0,3), eps));

    std::cout << "[PASS] SE3 exp/log<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_SEK3_exp_adj(T eps) {
    // SE_2(3) example: 6 dim -> K=1
    Eigen::Matrix<T, 6, 1> v;
    v << T(0.1), T(0.2), T(0.3), T(1.0), T(1.5), T(2.0);
    T dt = T(0.1);
    auto X = Exp_SEK3<T>(v, dt);
    // X should be (3+K) x (3+K) = 4x4
    assert(X.rows() == 4 && X.cols() == 4);
    // Check that the rotation part is orthogonal
    auto R = X.template block<3,3>(0,0);
    assert(isApprox(R * R.transpose(), Eigen::Matrix<T, 3, 3>::Identity(), eps));

    // Adjoint
    auto Adj = Adjoint_SEK3(X);
    assert(Adj.rows() == 6 && Adj.cols() == 6);
    // Ad(X) * Ad(X^{-1}) should be identity
    auto X_inv = X.inverse().eval();
    auto Adj_inv = Adjoint_SEK3(X_inv);
    assert(isApprox(Adj * Adj_inv, Eigen::Matrix<T, 6, 6>::Identity(), eps * T(10)));

    std::cout << "[PASS] SEK3 exp/adj<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_poly_solver() {
    // Solve x^3 - 6x^2 + 11x - 6 = 0  -> roots: 1, 2, 3
    Eigen::Matrix<T, 4, 1> coeff(T(1), T(-6), T(11), T(-6));
    auto roots = poly_solver_order3<T>(coeff);
    std::vector<T> expected = {T(1), T(2), T(3)};
    for (int i = 0; i < 3; ++i) {
        T diff = std::abs(roots[i].real() - expected[i]) + std::abs(roots[i].imag());
        assert(diff < T(1e-5));
    }

    // General solver
    Eigen::Matrix<T, 4, 1> coeff2 = coeff;
    auto roots2 = poly_solver<T>(coeff2);
    for (int i = 0; i < 3; ++i) {
        T diff = std::abs(roots2[i].real() - expected[i]) + std::abs(roots2[i].imag());
        assert(diff < T(1e-5));
    }

    std::cout << "[PASS] poly_solver<" << typeid(T).name() << ">" << std::endl;
}

template<typename T>
void test_dist_se3() {
    Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();
    Eigen::Matrix<T, 3, 1> t(T(1), T(0), T(0));
    T d = dist_se3<T>(R, t);
    assert(std::abs(d - T(1)) < T(1e-6));

    // Rotation of 90 deg around Z -> distance = pi/2
    R = Eigen::AngleAxis<T>(T(M_PI_2), Eigen::Matrix<T, 3, 1>::UnitZ()).toRotationMatrix();
    t.setZero();
    d = dist_se3<T>(R, t);
    assert(std::abs(d - T(M_PI_2)) < T(1e-6));

    std::cout << "[PASS] dist_se3<" << typeid(T).name() << ">" << std::endl;
}

int main() {
    std::cout << "Running LieGroup tests..." << std::endl;

    // Double precision tests
    test_skew_unskew<double>();
    test_hat2_wedge<double>();
    test_SO3_Jacobians<double>(EPS_D);
    test_SE3_Jacobians<double>(EPS_D);
    test_SO3_exp_log<double>(EPS_D);
    test_SE3_exp_log<double>(EPS_D);
    test_SEK3_exp_adj<double>(EPS_D);
    test_poly_solver<double>();
    test_dist_se3<double>();

    // Float precision tests
    test_skew_unskew<float>();
    test_hat2_wedge<float>();
    test_SO3_Jacobians<float>(EPS_F);
    test_SE3_Jacobians<float>(EPS_F);
    test_SO3_exp_log<float>(EPS_F);
    test_SE3_exp_log<float>(EPS_F);
    test_SEK3_exp_adj<float>(EPS_F);
    test_poly_solver<float>();
    test_dist_se3<float>();

    std::cout << "\nAll tests passed successfully!" << std::endl;
    return 0;
}
