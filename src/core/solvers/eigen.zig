//! Eigenvalue and eigenvector solvers
//!
//! Provides methods for computing eigenvalues and eigenvectors of matrices.
//! These are fundamental for:
//! - Principal Component Analysis (PCA)
//! - Stability analysis
//! - Vibration analysis
//! - Quantum mechanics
//!
//! Supported methods:
//! - Power iteration (largest eigenvalue/eigenvector)
//! - Inverse power iteration (smallest eigenvalue/eigenvector)
//! - QR algorithm (all eigenvalues)
//!
//! Future extensions:
//! - Jacobi method for symmetric matrices
//! - Arnoldi iteration for large sparse matrices
//! - Lanczos algorithm

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const vector = @import("../vector.zig");
const matrix = @import("../matrix.zig");
const solvers = @import("../solvers.zig");
const decomposition = @import("decomposition.zig");
const SolverError = solvers.SolverError;
const IterativeConfig = solvers.IterativeConfig;

/// Result of eigenvalue computation
pub fn EigenResult(comptime VecType: type) type {
    return struct {
        eigenvalue: f32,
        eigenvector: VecType,
        iterations: usize,
        converged: bool,
    };
}

/// Computes the dominant eigenvalue and eigenvector using power iteration
/// This finds the eigenvalue with the largest absolute value
///
/// Algorithm:
/// 1. Start with a random vector v
/// 2. Repeatedly compute v = A*v and normalize
/// 3. The ratio of consecutive components converges to the eigenvalue
pub fn powerIteration3x3(
    A: matrix.Mat3x3,
    config: IterativeConfig,
) !EigenResult(vector.Vec3) {
    // Start with a non-zero initial vector
    var v = vector.Vec3{ .x = 1.0, .y = 1.0, .z = 1.0 };
    v = v.normalize();

    var eigenvalue: f32 = 0.0;
    var iter: usize = 0;
    var converged = false;

    while (iter < config.max_iterations) : (iter += 1) {
        // Multiply by matrix: v_new = A * v
        const v_new = vector.Vec3{
            .x = A.x[0] * v.x + A.x[1] * v.y + A.x[2] * v.z,
            .y = A.y[0] * v.x + A.y[1] * v.y + A.y[2] * v.z,
            .z = A.z[0] * v.x + A.z[1] * v.y + A.z[2] * v.z,
        };

        // Compute the eigenvalue estimate (Rayleigh quotient)
        const new_eigenvalue = v_new.dot(&v) / v.dot(&v);

        // Check convergence
        if (iter > 0 and @abs(new_eigenvalue - eigenvalue) < config.tolerance) {
            eigenvalue = new_eigenvalue;
            v = v_new.normalize();
            converged = true;
            break;
        }

        eigenvalue = new_eigenvalue;
        v = v_new.normalize();

        if (config.verbose) {
            std.debug.print("Iteration {}: λ = {d:.6}\n", .{ iter, eigenvalue });
        }
    }

    return EigenResult(vector.Vec3){
        .eigenvalue = eigenvalue,
        .eigenvector = v,
        .iterations = iter,
        .converged = converged,
    };
}

/// Computes the smallest eigenvalue using inverse power iteration
/// This requires solving a linear system at each iteration
pub fn inversePowerIteration3x3(
    A: matrix.Mat3x3,
    config: IterativeConfig,
) !EigenResult(vector.Vec3) {
    // Compute LU decomposition once for efficiency
    const lu = try decomposition.lu3x3(A);

    // Start with a non-zero initial vector
    var v = vector.Vec3{ .x = 1.0, .y = 1.0, .z = 1.0 };
    v = v.normalize();

    var eigenvalue: f32 = 0.0;
    var iter: usize = 0;
    var converged = false;

    while (iter < config.max_iterations) : (iter += 1) {
        // Solve A * v_new = v (equivalent to v_new = A^(-1) * v)
        const v_new = try lu.solve(v);

        // Compute the eigenvalue estimate
        const new_eigenvalue = v.dot(&v_new) / v_new.dot(&v_new);

        // Check convergence
        if (iter > 0 and @abs(new_eigenvalue - eigenvalue) < config.tolerance) {
            eigenvalue = new_eigenvalue;
            v = v_new.normalize();
            converged = true;
            break;
        }

        eigenvalue = new_eigenvalue;
        v = v_new.normalize();

        if (config.verbose) {
            std.debug.print("Iteration {}: λ = {d:.6}\n", .{ iter, eigenvalue });
        }
    }

    // The eigenvalue of A is the reciprocal of the eigenvalue of A^(-1)
    eigenvalue = 1.0 / eigenvalue;

    return EigenResult(vector.Vec3){
        .eigenvalue = eigenvalue,
        .eigenvector = v,
        .iterations = iter,
        .converged = converged,
    };
}

/// Computes all eigenvalues using the QR algorithm
/// This iteratively applies QR decomposition to converge to a triangular form
///
/// Algorithm:
/// 1. A_0 = A
/// 2. For k = 0, 1, 2, ...
///    - Compute QR decomposition: A_k = Q_k * R_k
///    - Set A_(k+1) = R_k * Q_k
/// 3. A_k converges to upper triangular, with eigenvalues on diagonal
pub fn qrAlgorithm3x3(
    A: matrix.Mat3x3,
    config: IterativeConfig,
) !struct { eigenvalues: [3]f32, iterations: usize, converged: bool } {
    var A_k = A;
    var iter: usize = 0;
    var converged = false;

    while (iter < config.max_iterations) : (iter += 1) {
        // QR decomposition
        const qr = try decomposition.qr3x3(A_k);

        // A_(k+1) = R * Q
        A_k = qr.R.mul(&qr.Q);

        // Check convergence: off-diagonal elements should approach zero
        const off_diag_sum = @abs(A_k.x[1]) + @abs(A_k.x[2]) +
            @abs(A_k.y[0]) + @abs(A_k.y[2]) +
            @abs(A_k.z[0]) + @abs(A_k.z[1]);

        if (off_diag_sum < config.tolerance * 6) {
            converged = true;
            break;
        }

        if (config.verbose and iter % 10 == 0) {
            std.debug.print("Iteration {}: off-diagonal = {d:.6}\n", .{ iter, off_diag_sum });
        }
    }

    // Extract eigenvalues from diagonal
    const eigenvalues = [3]f32{ A_k.x[0], A_k.y[1], A_k.z[2] };

    return .{
        .eigenvalues = eigenvalues,
        .iterations = iter,
        .converged = converged,
    };
}

/// Computes the Rayleigh quotient: (v^T * A * v) / (v^T * v)
/// This provides an estimate of an eigenvalue given an approximate eigenvector
pub fn rayleighQuotient3x3(A: matrix.Mat3x3, v: vector.Vec3) f32 {
    const Av = vector.Vec3{
        .x = A.x[0] * v.x + A.x[1] * v.y + A.x[2] * v.z,
        .y = A.y[0] * v.x + A.y[1] * v.y + A.y[2] * v.z,
        .z = A.z[0] * v.x + A.z[1] * v.y + A.z[2] * v.z,
    };
    return v.dot(&Av) / v.dot(&v);
}

// Tests

test "power iteration finds dominant eigenvalue" {
    // Matrix with known eigenvalues: 5, 2, 1
    // [4  1  0]
    // [1  3  1]
    // [0  1  2]
    const A = matrix.Mat3x3{
        .x = .{ 4.0, 1.0, 0.0 },
        .y = .{ 1.0, 3.0, 1.0 },
        .z = .{ 0.0, 1.0, 2.0 },
    };

    const config = IterativeConfig{
        .max_iterations = 1000,
        .tolerance = 1e-6,
        .verbose = false,
    };

    const result = try powerIteration3x3(A, config);

    try std.testing.expect(result.converged);
    // Dominant eigenvalue should be approximately 5
    try std.testing.expectApproxEqAbs(5.0, result.eigenvalue, 0.1);
}

test "inverse power iteration finds smallest eigenvalue" {
    // Same matrix as above, smallest eigenvalue should be approximately 1
    const A = matrix.Mat3x3{
        .x = .{ 4.0, 1.0, 0.0 },
        .y = .{ 1.0, 3.0, 1.0 },
        .z = .{ 0.0, 1.0, 2.0 },
    };

    const config = IterativeConfig{
        .max_iterations = 1000,
        .tolerance = 1e-6,
        .verbose = false,
    };

    const result = try inversePowerIteration3x3(A, config);

    try std.testing.expect(result.converged);
    // Smallest eigenvalue should be approximately 1
    try std.testing.expectApproxEqAbs(1.0, result.eigenvalue, 0.1);
}

test "QR algorithm finds all eigenvalues" {
    // Symmetric matrix for easier testing
    // [3  1  0]
    // [1  3  1]
    // [0  1  3]
    // Eigenvalues: 3-√2 ≈ 1.586, 3, 3+√2 ≈ 4.414
    const A = matrix.Mat3x3{
        .x = .{ 3.0, 1.0, 0.0 },
        .y = .{ 1.0, 3.0, 1.0 },
        .z = .{ 0.0, 1.0, 3.0 },
    };

    const config = IterativeConfig{
        .max_iterations = 1000,
        .tolerance = 1e-4,
        .verbose = false,
    };

    const result = try qrAlgorithm3x3(A, config);

    try std.testing.expect(result.converged);

    // Sort eigenvalues for easier testing
    var eigs = result.eigenvalues;
    std.mem.sort(f32, &eigs, {}, comptime std.sort.asc(f32));

    // Check eigenvalues (with reasonable tolerance due to numerical errors)
    try std.testing.expectApproxEqAbs(1.586, eigs[0], 0.05);
    try std.testing.expectApproxEqAbs(3.0, eigs[1], 0.05);
    try std.testing.expectApproxEqAbs(4.414, eigs[2], 0.05);
}

test "Rayleigh quotient" {
    const A = matrix.Mat3x3{
        .x = .{ 2.0, 0.0, 0.0 },
        .y = .{ 0.0, 3.0, 0.0 },
        .z = .{ 0.0, 0.0, 4.0 },
    };

    // Eigenvector corresponding to eigenvalue 3
    const v = vector.Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const rq = rayleighQuotient3x3(A, v);

    try std.testing.expectApproxEqAbs(3.0, rq, 1e-5);
}
