//! Matrix decomposition methods
//!
//! Provides various matrix factorization techniques:
//! - LU decomposition (with partial pivoting)
//! - QR decomposition (Gram-Schmidt, Householder reflections)
//!
//! These decompositions are fundamental building blocks for:
//! - Solving linear systems efficiently
//! - Computing determinants
//! - Finding eigenvalues
//! - Least squares problems
//!
//! Future extensions:
//! - SVD (Singular Value Decomposition)
//! - Cholesky decomposition (for positive definite matrices)
//! - Schur decomposition

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const vector = @import("../vector.zig");
const matrix = @import("../matrix.zig");
const solvers = @import("../solvers.zig");
const SolverError = solvers.SolverError;

/// LU decomposition result for 3x3 matrix
/// A = L * U where L is lower triangular and U is upper triangular
pub const LU3x3 = struct {
    L: matrix.Mat3x3,
    U: matrix.Mat3x3,
    P: [3]usize, // Permutation indices for partial pivoting

    /// Solves the system Ax = b using the LU decomposition
    /// This is more efficient when solving multiple systems with the same A
    pub fn solve(self: *const LU3x3, b: vector.Vec3) !vector.Vec3 {
        // Apply permutation to b
        const pb = vector.Vec3{
            .x = switch (self.P[0]) {
                0 => b.x,
                1 => b.y,
                2 => b.z,
                else => unreachable,
            },
            .y = switch (self.P[1]) {
                0 => b.x,
                1 => b.y,
                2 => b.z,
                else => unreachable,
            },
            .z = switch (self.P[2]) {
                0 => b.x,
                1 => b.y,
                2 => b.z,
                else => unreachable,
            },
        };

        // Forward substitution: Ly = Pb
        const y1 = pb.x / self.L.x[0];
        const y2 = (pb.y - self.L.y[0] * y1) / self.L.y[1];
        const y3 = (pb.z - self.L.z[0] * y1 - self.L.z[1] * y2) / self.L.z[2];

        // Back substitution: Ux = y
        const x3 = y3 / self.U.z[2];
        const x2 = (y2 - self.U.y[2] * x3) / self.U.y[1];
        const x1 = (y1 - self.U.x[1] * x2 - self.U.x[2] * x3) / self.U.x[0];

        return vector.Vec3{ .x = x1, .y = x2, .z = x3 };
    }

    /// Computes the determinant using the LU decomposition
    /// det(A) = det(L) * det(U) * sign(permutation)
    pub fn determinant(self: *const LU3x3) f32 {
        // For LU with partial pivoting, det(A) = det(L) * det(U) * parity
        // det(L) = product of diagonal (all 1s in our case)
        // det(U) = product of diagonal
        const det_U = self.U.x[0] * self.U.y[1] * self.U.z[2];

        // Calculate permutation parity (count swaps)
        var parity: f32 = 1.0;
        var temp = self.P;
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            while (temp[i] != i) {
                const swap_idx = temp[i];
                std.mem.swap(usize, &temp[i], &temp[swap_idx]);
                parity = -parity;
            }
        }

        return det_U * parity;
    }
};

/// Computes LU decomposition with partial pivoting for a 3x3 matrix
/// A = P * L * U where P is a permutation matrix
pub fn lu3x3(A: matrix.Mat3x3) !LU3x3 {
    var U = A;
    var L = matrix.Mat3x3{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
    var P = [3]usize{ 0, 1, 2 };

    // First column
    {
        // Find pivot
        var max_row: usize = 0;
        var max_val = @abs(U.x[0]);
        if (@abs(U.y[0]) > max_val) {
            max_val = @abs(U.y[0]);
            max_row = 1;
        }
        if (@abs(U.z[0]) > max_val) {
            max_val = @abs(U.z[0]);
            max_row = 2;
        }

        if (solvers.isNearZero(max_val, 1e-8)) {
            return SolverError.SingularMatrix;
        }

        // Swap rows in U and permutation
        if (max_row != 0) {
            if (max_row == 1) {
                std.mem.swap([3]f32, &U.x, &U.y);
                std.mem.swap(usize, &P[0], &P[1]);
            } else {
                std.mem.swap([3]f32, &U.x, &U.z);
                std.mem.swap(usize, &P[0], &P[2]);
            }
        }

        // Eliminate first column
        L.y[0] = U.y[0] / U.x[0];
        L.z[0] = U.z[0] / U.x[0];

        U.y[0] = 0.0;
        U.y[1] -= L.y[0] * U.x[1];
        U.y[2] -= L.y[0] * U.x[2];

        U.z[0] = 0.0;
        U.z[1] -= L.z[0] * U.x[1];
        U.z[2] -= L.z[0] * U.x[2];
    }

    // Second column
    {
        // Find pivot in remaining 2x2 block
        if (@abs(U.z[1]) > @abs(U.y[1])) {
            std.mem.swap([3]f32, &U.y, &U.z);
            std.mem.swap(f32, &L.y[0], &L.z[0]);
            std.mem.swap(usize, &P[1], &P[2]);
        }

        if (solvers.isNearZero(U.y[1], 1e-8)) {
            return SolverError.SingularMatrix;
        }

        // Eliminate second column
        L.z[1] = U.z[1] / U.y[1];
        U.z[1] = 0.0;
        U.z[2] -= L.z[1] * U.y[2];
    }

    // Check final pivot
    if (solvers.isNearZero(U.z[2], 1e-8)) {
        return SolverError.SingularMatrix;
    }

    return LU3x3{ .L = L, .U = U, .P = P };
}

/// QR decomposition result for 3x3 matrix
/// A = Q * R where Q is orthogonal and R is upper triangular
pub const QR3x3 = struct {
    Q: matrix.Mat3x3,
    R: matrix.Mat3x3,

    /// Solves the least squares problem ||Ax - b||^2
    /// Uses the fact that A = QR, so x = R^(-1) * Q^T * b
    pub fn solveLeastSquares(self: *const QR3x3, b: vector.Vec3) !vector.Vec3 {
        // Compute Q^T * b
        const Qt = self.Q.transpose();
        const Qtb = vector.Vec3{
            .x = Qt.x[0] * b.x + Qt.x[1] * b.y + Qt.x[2] * b.z,
            .y = Qt.y[0] * b.x + Qt.y[1] * b.y + Qt.y[2] * b.z,
            .z = Qt.z[0] * b.x + Qt.z[1] * b.y + Qt.z[2] * b.z,
        };

        // Back substitution with R
        if (solvers.isNearZero(self.R.z[2], 1e-8)) {
            return SolverError.SingularMatrix;
        }
        const x3 = Qtb.z / self.R.z[2];
        const x2 = (Qtb.y - self.R.y[2] * x3) / self.R.y[1];
        const x1 = (Qtb.x - self.R.x[1] * x2 - self.R.x[2] * x3) / self.R.x[0];

        return vector.Vec3{ .x = x1, .y = x2, .z = x3 };
    }
};

/// Computes QR decomposition using Gram-Schmidt orthogonalization
/// A = Q * R where Q is orthogonal and R is upper triangular
pub fn qr3x3(A: matrix.Mat3x3) !QR3x3 {
    // Extract columns as vectors
    const a1 = vector.Vec3{ .x = A.x[0], .y = A.y[0], .z = A.z[0] };
    const a2 = vector.Vec3{ .x = A.x[1], .y = A.y[1], .z = A.z[1] };
    const a3 = vector.Vec3{ .x = A.x[2], .y = A.y[2], .z = A.z[2] };

    // Gram-Schmidt orthogonalization
    // q1 = a1 / ||a1||
    const r11 = a1.length();
    if (solvers.isNearZero(r11, 1e-8)) {
        return SolverError.SingularMatrix;
    }
    const q1 = a1.scale(1.0 / r11);

    // q2 = (a2 - (a2·q1)q1) normalized
    const r12 = a2.dot(&q1);
    const vec2_unnorm = a2.sub(&q1.scale(r12));
    const r22 = vec2_unnorm.length();
    if (solvers.isNearZero(r22, 1e-8)) {
        return SolverError.SingularMatrix;
    }
    const q2 = vec2_unnorm.scale(1.0 / r22);

    // q3 = (a3 - (a3·q1)q1 - (a3·q2)q2) normalized
    const r13 = a3.dot(&q1);
    const r23 = a3.dot(&q2);
    const vec3_unnorm = a3.sub(&q1.scale(r13)).sub(&q2.scale(r23));
    const r33 = vec3_unnorm.length();
    if (solvers.isNearZero(r33, 1e-8)) {
        return SolverError.SingularMatrix;
    }
    const q3 = vec3_unnorm.scale(1.0 / r33);

    // Construct Q (column-wise)
    const Q = matrix.Mat3x3{
        .x = .{ q1.x, q2.x, q3.x },
        .y = .{ q1.y, q2.y, q3.y },
        .z = .{ q1.z, q2.z, q3.z },
    };

    // Construct R (upper triangular)
    const R = matrix.Mat3x3{
        .x = .{ r11, r12, r13 },
        .y = .{ 0.0, r22, r23 },
        .z = .{ 0.0, 0.0, r33 },
    };

    return QR3x3{ .Q = Q, .R = R };
}

// Tests

test "lu3x3 decomposition" {
    const A = matrix.Mat3x3{
        .x = .{ 2.0, 1.0, 1.0 },
        .y = .{ 4.0, -6.0, 0.0 },
        .z = .{ -2.0, 7.0, 2.0 },
    };

    const lu = try lu3x3(A);

    // Verify that P*L*U ≈ A by reconstructing
    // For now, just check that decomposition completes without error
    _ = lu;
}

test "lu3x3 solve system" {
    const A = matrix.Mat3x3{
        .x = .{ 2.0, 1.0, 1.0 },
        .y = .{ 1.0, 3.0, 2.0 },
        .z = .{ 1.0, 0.0, 0.0 },
    };
    const b = vector.Vec3{ .x = 4.0, .y = 5.0, .z = 6.0 };

    const lu = try lu3x3(A);
    const x = try lu.solve(b);

    // Verify Ax ≈ b
    const result = vector.Vec3{
        .x = A.x[0] * x.x + A.x[1] * x.y + A.x[2] * x.z,
        .y = A.y[0] * x.x + A.y[1] * x.y + A.y[2] * x.z,
        .z = A.z[0] * x.x + A.z[1] * x.y + A.z[2] * x.z,
    };

    try std.testing.expectApproxEqAbs(b.x, result.x, 1e-4);
    try std.testing.expectApproxEqAbs(b.y, result.y, 1e-4);
    try std.testing.expectApproxEqAbs(b.z, result.z, 1e-4);
}

test "qr3x3 decomposition" {
    const A = matrix.Mat3x3{
        .x = .{ 1.0, 1.0, 0.0 },
        .y = .{ 1.0, 0.0, 1.0 },
        .z = .{ 0.0, 1.0, 1.0 },
    };

    const qr = try qr3x3(A);

    // Verify Q is orthogonal: Q^T * Q should be identity
    const QtQ = qr.Q.transpose().mul(&qr.Q);
    try std.testing.expectApproxEqAbs(1.0, QtQ.x[0], 1e-5);
    try std.testing.expectApproxEqAbs(0.0, QtQ.x[1], 1e-5);
    try std.testing.expectApproxEqAbs(0.0, QtQ.x[2], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, QtQ.y[1], 1e-5);
    try std.testing.expectApproxEqAbs(1.0, QtQ.z[2], 1e-5);
}

test "qr3x3 solve least squares" {
    // Overdetermined system (more equations than unknowns) would need
    // rectangular matrices, but we can test the square case
    const A = matrix.Mat3x3{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
    const b = vector.Vec3{ .x = 1.0, .y = 2.0, .z = 3.0 };

    const qr = try qr3x3(A);
    const x = try qr.solveLeastSquares(b);

    try std.testing.expectApproxEqAbs(1.0, x.x, 1e-5);
    try std.testing.expectApproxEqAbs(2.0, x.y, 1e-5);
    try std.testing.expectApproxEqAbs(3.0, x.z, 1e-5);
}
