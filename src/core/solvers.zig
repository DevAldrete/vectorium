//! Linear algebra solvers module
//!
//! This module provides a flexible and extensible architecture for solving
//! various linear algebra problems. The design follows these principles:
//!
//! 1. Separation of concerns: Different solver types are organized into submodules
//! 2. Strategy pattern: Multiple algorithms for the same problem type
//! 3. Generics: Support for different matrix dimensions and numeric types
//! 4. Evolvability: Easy to add new solvers and extend existing ones
//!
//! Supported solver categories:
//! - Linear systems (Ax = b)
//! - Eigenvalue/eigenvector problems
//! - Matrix decompositions (LU, QR, SVD, etc.)
//! - Least squares problems

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const vector = @import("vector.zig");
const matrix = @import("matrix.zig");

// Re-export submodules for organized access
pub const linear_systems = @import("solvers/linear_systems.zig");
pub const eigen = @import("solvers/eigen.zig");
pub const decomposition = @import("solvers/decomposition.zig");

// Common error types for all solvers
pub const SolverError = error{
    SingularMatrix,
    NoConvergence,
    InvalidInput,
    NumericalInstability,
    DimensionMismatch,
};

// Configuration for iterative solvers
pub const IterativeConfig = struct {
    max_iterations: usize = 1000,
    tolerance: f32 = 1e-6,
    verbose: bool = false,
};

// Result type for iterative methods
pub fn IterativeResult(comptime T: type) type {
    return struct {
        solution: T,
        iterations: usize,
        residual: f32,
        converged: bool,
    };
}

// Numerical tolerance utilities
pub fn isNearZero(value: f32, epsilon: f32) bool {
    return @abs(value) < epsilon;
}

pub fn isClose(a: f32, b: f32, rel_tol: f32, abs_tol: f32) bool {
    const diff = @abs(a - b);
    return diff <= @max(rel_tol * @max(@abs(a), @abs(b)), abs_tol);
}

// Helper to check if a matrix is square (compile-time when possible)
pub fn assertSquareMatrix(rows: usize, cols: usize) !void {
    if (rows != cols) {
        return SolverError.DimensionMismatch;
    }
}

test "isNearZero" {
    try std.testing.expect(isNearZero(1e-7, 1e-6));
    try std.testing.expect(!isNearZero(1e-5, 1e-6));
}

test "isClose" {
    try std.testing.expect(isClose(1.0, 1.0000001, 1e-5, 1e-8));
    try std.testing.expect(!isClose(1.0, 1.001, 1e-5, 1e-8));
}
