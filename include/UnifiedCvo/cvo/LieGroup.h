#ifndef __LIEGROUP_H__
#define __LIEGROUP_H__

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <complex>

namespace cvo {

template <typename Derived>
using ScalarT = typename Derived::Scalar;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
template <typename T>
constexpr T tolerance() { return T(1e-6); }

// -----------------------------------------------------------------------------
// skew / unskew
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 3, RC_MAJOR> skew(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3, RC_MAJOR> M = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Zero();
    M << T(0), -v[2],  v[1],
         v[2],  T(0), -v[0],
        -v[1],  v[0],  T(0);
    return M;
}

template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 1> unskew(const Eigen::Matrix<T, 3, 3, RC_MAJOR>& M) {
    Eigen::Matrix<T, 3, 1> v;
    v << M(2,1), M(0,2), M(1,0);
    return v;
}

template <typename Derived>
Eigen::Matrix<ScalarT<Derived>, 3, 1> unskew(const Eigen::MatrixBase<Derived>& M) {
    Eigen::Matrix<ScalarT<Derived>, 3, 1> v;
    v << M(2,1), M(0,2), M(1,0);
    return v;
}

// -----------------------------------------------------------------------------
// hat2 / wedge (SE(3) ↔ se(3))
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 4, 4, RC_MAJOR> hat2(const Eigen::Matrix<T, 6, 1>& x) {
    Eigen::Matrix<T, 4, 4, RC_MAJOR> X = Eigen::Matrix<T, 4, 4, RC_MAJOR>::Zero();
    X.template block<3,3>(0,0) = skew<T, RC_MAJOR>(x.template head<3>());
    X.template block<3,1>(0,3) = x.template tail<3>();
    return X;
}

template <typename T>
Eigen::Matrix<T, 6, 1> wedge(const Eigen::Matrix<T, 4, 4>& X) {
    Eigen::Matrix<T, 6, 1> x;
    x.template head<3>() = unskew(X.template block<3,3>(0,0));
    x.template tail<3>() = X.template block<3,1>(0,3);
    return x;
}

template <typename Derived>
Eigen::Matrix<ScalarT<Derived>, 6, 1> wedge(const Eigen::MatrixBase<Derived>& X) {
    Eigen::Matrix<ScalarT<Derived>, 6, 1> x;
    x.template head<3>() = unskew(X.template block<3,3>(0,0));
    x.template tail<3>() = X.template block<3,1>(0,3);
    return x;
}

// -----------------------------------------------------------------------------
// Forward declarations
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> Adjoint_SEK3(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>& X);

template <typename Derived>
Eigen::Matrix<ScalarT<Derived>, Eigen::Dynamic, Eigen::Dynamic> Adjoint_SEK3(
    const Eigen::MatrixBase<Derived>& X);

// -----------------------------------------------------------------------------
// Left Jacobian of SO(3)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 3, RC_MAJOR> LeftJacobian_SO3(const Eigen::Matrix<T, 3, 1>& w) {
    Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
    T theta = w.norm();
    if (theta < tolerance<T>()) {
        return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
    }
    T theta2 = theta * theta;
    T theta3 = theta2 * theta;
    return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity()
           + ((T(1) - std::cos(theta)) / theta2) * A
           + ((theta - std::sin(theta)) / theta3) * (A * A);
}

template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 3, RC_MAJOR> LeftJacobianInverse_SO3(const Eigen::Matrix<T, 3, 1>& w) {
    Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
    T theta = w.norm();
    if (theta < tolerance<T>()) {
        return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
    }
    T theta2 = theta * theta;
    T sin_theta = std::sin(theta);
    T cos_theta = std::cos(theta);
    return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity()
           - T(0.5) * A
           + (T(1) / theta2 - (T(1) + cos_theta) / (T(2) * theta * sin_theta)) * (A * A);
}

// -----------------------------------------------------------------------------
// Left / Right Jacobians of SE(3)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 6, 6, RC_MAJOR> LeftJacobian_SE3(const Eigen::Matrix<T, 6, 1>& v) {
    Eigen::Matrix<T, 6, 6, RC_MAJOR> J = Eigen::Matrix<T, 6, 6, RC_MAJOR>::Zero();
    Eigen::Matrix<T, 3, 1> Phi = v.template head<3>();
    Eigen::Matrix<T, 3, 1> Rho = v.template tail<3>();
    T phi = Phi.norm();
    Eigen::Matrix<T, 3, 3, RC_MAJOR> Phi_skew = skew<T, RC_MAJOR>(Phi);
    Eigen::Matrix<T, 3, 3, RC_MAJOR> Rho_skew = skew<T, RC_MAJOR>(Rho);
    Eigen::Matrix<T, 3, 3, RC_MAJOR> J_so3 = LeftJacobian_SO3<T, RC_MAJOR>(Phi);
    Eigen::Matrix<T, 3, 3, RC_MAJOR> Q = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Zero();

    if (phi < tolerance<T>()) {
        Q = T(0.5) * Rho_skew;
    } else {
        T phi2 = phi * phi;
        T phi3 = phi2 * phi;
        T phi4 = phi3 * phi;
        T phi5 = phi4 * phi;
        T sin_phi = std::sin(phi);
        T cos_phi = std::cos(phi);
        Q = T(0.5) * Rho_skew
            + (phi - sin_phi) / phi3 * (Phi_skew * Rho_skew + Rho_skew * Phi_skew + Phi_skew * Rho_skew * Phi_skew)
            - (T(1) - T(0.5) * phi2 - cos_phi) / phi4 * (Phi_skew * Phi_skew * Rho_skew + Rho_skew * Phi_skew * Phi_skew
                                                         - T(3) * Phi_skew * Rho_skew * Phi_skew)
            - T(0.5) * ((T(1) - T(0.5) * phi2 - cos_phi) / phi4
                        - T(3) * (phi - sin_phi - phi3 / T(6)) / phi5)
              * (Phi_skew * Rho_skew * Phi_skew * Phi_skew + Phi_skew * Phi_skew * Rho_skew * Phi_skew);
    }
    J.template block<3,3>(0,0) = J_so3;
    J.template block<3,3>(0,3) = Q;
    J.template block<3,3>(3,3) = J_so3;
    return J;
}

template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 6, 6, RC_MAJOR> RightJacobian_SE3(const Eigen::Matrix<T, 6, 1>& v) {
    return LeftJacobian_SE3<T, RC_MAJOR>(-v);
}

template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 6, 6, RC_MAJOR> RightJacobianInverse_SE3(const Eigen::Matrix<T, 6, 1>& v) {
    if (v.norm() < tolerance<T>())
        return Eigen::Matrix<T, 6, 6, RC_MAJOR>::Identity();
    Eigen::Matrix<T, 6, 6, RC_MAJOR> Jr = RightJacobian_SE3<T, RC_MAJOR>(v);
    return Jr.inverse();
}

// -----------------------------------------------------------------------------
// Exponential map: SO(3)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 3, RC_MAJOR> Exp_SO3(const Eigen::Matrix<T, 3, 1>& w) {
    Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
    T theta = w.norm();
    if (theta < tolerance<T>()) {
        return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
    }
    T sin_theta = std::sin(theta);
    T cos_theta = std::cos(theta);
    return Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity()
           + (sin_theta / theta) * A
           + ((T(1) - cos_theta) / (theta * theta)) * (A * A);
}

// -----------------------------------------------------------------------------
// Logarithm map: SO(3)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 1> Log_SO3(const Eigen::Matrix<T, 3, 3, RC_MAJOR>& R) {
    const T cos_theta = std::max(T(-1), std::min(T(1), (R.trace() - T(1)) / T(2)));
    T theta = std::acos(cos_theta);
    if (theta < tolerance<T>()) {
        return Eigen::Matrix<T, 3, 1>::Zero();
    }
    return unskew<T, RC_MAJOR>(theta / (T(2) * std::sin(theta)) * (R - R.transpose()));
}

// -----------------------------------------------------------------------------
// Exponential / Logarithm for SE(3) (returns 4x4 matrix)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 4, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v) {
    Eigen::Matrix<T, 4, 4, RC_MAJOR> X = Eigen::Matrix<T, 4, 4, RC_MAJOR>::Identity();
    X.template block<3,3>(0,0) = Exp_SO3<T, RC_MAJOR>(v.template head<3>());
    X.template block<3,1>(0,3) = LeftJacobian_SO3<T, RC_MAJOR>(v.template head<3>()) * v.template tail<3>();
    return X;
}

template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 6, 1> Log_SE3(const Eigen::Matrix<T, 4, 4, RC_MAJOR>& X) {
    Eigen::Matrix<T, 3, 1> w = Log_SO3<T, RC_MAJOR>(X.template block<3,3>(0,0));
    Eigen::Matrix<T, 3, 1> u = LeftJacobianInverse_SO3<T, RC_MAJOR>(w) * X.template block<3,1>(0,3);
    Eigen::Matrix<T, 6, 1> xi;
    xi << w, u;
    return xi;
}

// -----------------------------------------------------------------------------
// Exponential map for SE(3) that returns a 3x4 matrix (rotation + translation)
// (with optional flag to swap w / u order)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, 3, 4, RC_MAJOR> Exp_SE3(const Eigen::Matrix<T, 6, 1>& v, bool is_wu) {
    Eigen::Matrix<T, 3, 1> w, u;
    if (is_wu) {
        w = v.template head<3>();
        u = v.template tail<3>();
    } else {
        w = v.template tail<3>();
        u = v.template head<3>();
    }
    Eigen::Matrix<T, 3, 4, RC_MAJOR> X;
    X.template block<3,3>(0,0) = Exp_SO3<T, RC_MAJOR>(w);
    X.template block<3,1>(0,3) = LeftJacobian_SO3<T, RC_MAJOR>(w) * u;
    return X;
}

// -----------------------------------------------------------------------------
// Exponential map for SE_K(3) (generalised pose with multiple velocities)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> Exp_SEK3(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v, T dt) {
    int K = (v.size() - 3) / 3;
    int dim = 3 + K;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> X = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>::Identity(dim, dim);
    Eigen::Matrix<T, 3, 1> w = v.template head<3>();
    T theta = w.norm();
    Eigen::Matrix<T, 3, 3, RC_MAJOR> R, Jl;
    if (theta < tolerance<T>()) {
        R = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
        Jl = dt * Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity();
    } else {
        Eigen::Matrix<T, 3, 3, RC_MAJOR> A = skew<T, RC_MAJOR>(w);
        T theta2 = theta * theta;
        T stheta = std::sin(dt * theta);
        T ctheta = std::cos(dt * theta);
        T oneMinusCosTheta2 = (T(1) - ctheta) / theta2;
        Eigen::Matrix<T, 3, 3, RC_MAJOR> A2 = A * A;
        R = Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity()
            + (stheta / theta) * A
            + oneMinusCosTheta2 * A2;
        Jl = dt * Eigen::Matrix<T, 3, 3, RC_MAJOR>::Identity()
             + oneMinusCosTheta2 * A
             + ((dt * theta - stheta) / (theta2 * theta)) * A2;
    }
    X.template block<3,3>(0,0) = R;
    for (int i = 0; i < K; ++i) {
        X.template block<3,1>(0, 3 + i) = Jl * v.template segment<3>(3 + 3 * i);
    }
    return X;
}

// -----------------------------------------------------------------------------
// Adjoint for SE_K(3)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> Adjoint_SEK3(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>& X) {
    int K = X.cols() - 3;
    int dim = 3 + 3 * K;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR> Adj = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RC_MAJOR>::Zero(dim, dim);
    Eigen::Matrix<T, 3, 3, RC_MAJOR> R = X.template block<3,3>(0,0);
    Adj.template block<3,3>(0,0) = R;
    for (int i = 0; i < K; ++i) {
        Adj.template block<3,3>(3 + 3 * i, 3 + 3 * i) = R;
        Adj.template block<3,3>(3 + 3 * i, 0) = skew<T, RC_MAJOR>(X.template block<3,1>(0, 3 + i)) * R;
    }
    return Adj;
}

template <typename Derived>
Eigen::Matrix<ScalarT<Derived>, Eigen::Dynamic, Eigen::Dynamic> Adjoint_SEK3(
    const Eigen::MatrixBase<Derived>& X) {
    using T = ScalarT<Derived>;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X_eval = X.eval();
    return Adjoint_SEK3<T, Eigen::ColMajor>(X_eval);
}

// -----------------------------------------------------------------------------
// Polynomial solver (companion matrix method)
// -----------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> poly_solver(const Eigen::Matrix<T, Eigen::Dynamic, 1>& coef) {
    int order = coef.size() - 1;
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    MatrixT M = MatrixT::Zero(order, order);
    M.bottomLeftCorner(order - 1, order - 1) = MatrixT::Identity(order - 1, order - 1);
    for (int i = 0; i < order; ++i) {
        M(0, i) = -coef(i + 1) / coef(0);
    }
    return M.template cast<std::complex<T>>().eigenvalues();
}

template <typename T>
Eigen::Matrix<std::complex<T>, 3, 1> poly_solver_order3(const Eigen::Matrix<T, 4, 1>& coef) {
    Eigen::Matrix<T, 3, 3> M = Eigen::Matrix<T, 3, 3>::Zero();
    M.template bottomLeftCorner<2,2>().setIdentity();
    for (int i = 0; i < 3; ++i) {
        M(0, i) = -coef(i + 1) / coef(0);
    }
    return M.template cast<std::complex<T>>().eigenvalues();
}

// -----------------------------------------------------------------------------
// Distance on SE(3) (Frobenius norm of log)
// -----------------------------------------------------------------------------
template <typename T, int RC_MAJOR = Eigen::ColMajor>
T dist_se3(const Eigen::Matrix<T, 3, 3, RC_MAJOR>& R, const Eigen::Matrix<T, 3, 1>& t) {
    const T rot = Log_SO3<T, RC_MAJOR>(R).norm();
    const T trans = t.norm();
    return std::sqrt(rot * rot + trans * trans);
}

} // namespace cvo

#endif // __LIEGROUP_H__
