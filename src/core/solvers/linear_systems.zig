//! Linear system solvers
//!
//! Solves systems of linear equations of the form Ax = b
//! where A is an n×n matrix, b is an n-dimensional vector,
//! and x is the unknown vector to solve for.
//!
//! Supported methods:
//! - Gaussian elimination with partial pivoting (direct)
//! - Back substitution (for upper triangular systems)
//! - Forward substitution (for lower triangular systems)
//!
//! Future extensions:
//! - Iterative methods (Jacobi, Gauss-Seidel, Conjugate Gradient)
//! - Sparse matrix solvers
//! - Overdetermined/underdetermined systems

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const vector = @import("../vector.zig");
const matrix = @import("../matrix.zig");
const solvers = @import("../solvers.zig");
const SolverError = solvers.SolverError;

/// Solves a 2x2 linear system using Gaussian elimination
/// A * x = b
pub fn solve2x2(A: matrix.Mat2x2, b: vector.Vec2) !vector.Vec2 {
    // Make a copy to work with
    var work_matrix = A;
    var work_vector = [2]f32{ b.x, b.y };

    // Forward elimination with partial pivoting
    if (@abs(work_matrix.x[0]) < @abs(work_matrix.y[0])) {
        // Swap rows
        std.mem.swap([2]f32, &work_matrix.x, &work_matrix.y);
        std.mem.swap(f32, &work_vector[0], &work_vector[1]);
    }

    // Check for singular matrix
    if (solvers.isNearZero(work_matrix.x[0], 1e-8)) {
        return SolverError.SingularMatrix;
    }

    // Eliminate first column of second row
    const factor = work_matrix.y[0] / work_matrix.x[0];
    work_matrix.y[0] = 0.0;
    work_matrix.y[1] -= factor * work_matrix.x[1];
    work_vector[1] -= factor * work_vector[0];

    // Check if second pivot is zero
    if (solvers.isNearZero(work_matrix.y[1], 1e-8)) {
        return SolverError.SingularMatrix;
    }

    // Back substitution
    const y = work_vector[1] / work_matrix.y[1];
    const x = (work_vector[0] - work_matrix.x[1] * y) / work_matrix.x[0];

    return vector.Vec2{ .x = x, .y = y };
}

/// Solves a 3x3 linear system using Gaussian elimination with partial pivoting
/// A * x = b
pub fn solve3x3(A: matrix.Mat3x3, b: vector.Vec3) !vector.Vec3 {
    var work_matrix = A;
    var work_vector = [3]f32{ b.x, b.y, b.z };

    // Forward elimination with partial pivoting
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        // Find pivot
        var max_row = i;
        var max_val = @abs(getMatrixElement3x3(&work_matrix, i, i));

        var k: usize = i + 1;
        while (k < 3) : (k += 1) {
            const val = @abs(getMatrixElement3x3(&work_matrix, k, i));
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        // Check for singular matrix
        if (solvers.isNearZero(max_val, 1e-8)) {
            return SolverError.SingularMatrix;
        }

        // Swap rows if needed
        if (max_row != i) {
            swapRows3x3(&work_matrix, i, max_row);
            std.mem.swap(f32, &work_vector[i], &work_vector[max_row]);
        }

        // Eliminate column below pivot
        var j: usize = i + 1;
        while (j < 3) : (j += 1) {
            const factor = getMatrixElement3x3(&work_matrix, j, i) /
                getMatrixElement3x3(&work_matrix, i, i);

            var col: usize = i;
            while (col < 3) : (col += 1) {
                const current = getMatrixElement3x3(&work_matrix, j, col);
                const pivot = getMatrixElement3x3(&work_matrix, i, col);
                setMatrixElement3x3(&work_matrix, j, col, current - factor * pivot);
            }
            work_vector[j] -= factor * work_vector[i];
        }
    }

    // Check final diagonal element
    if (solvers.isNearZero(work_matrix.z[2], 1e-8)) {
        return SolverError.SingularMatrix;
    }

    // Back substitution
    const z = work_vector[2] / work_matrix.z[2];
    const y = (work_vector[1] - work_matrix.y[2] * z) / work_matrix.y[1];
    const x = (work_vector[0] - work_matrix.x[1] * y - work_matrix.x[2] * z) / work_matrix.x[0];

    return vector.Vec3{ .x = x, .y = y, .z = z };
}

/// Solves a 4x4 linear system using Gaussian elimination with partial pivoting
/// A * x = b
pub fn solve4x4(A: matrix.Mat4x4, b: vector.Vec4) !vector.Vec4 {
    var work_matrix = A;
    var work_vector = [4]f32{ b.x, b.y, b.z, b.w };

    // Forward elimination with partial pivoting
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        // Find pivot
        var max_row = i;
        var max_val = @abs(getMatrixElement4x4(&work_matrix, i, i));

        var k: usize = i + 1;
        while (k < 4) : (k += 1) {
            const val = @abs(getMatrixElement4x4(&work_matrix, k, i));
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }

        // Check for singular matrix
        if (solvers.isNearZero(max_val, 1e-8)) {
            return SolverError.SingularMatrix;
        }

        // Swap rows if needed
        if (max_row != i) {
            swapRows4x4(&work_matrix, i, max_row);
            std.mem.swap(f32, &work_vector[i], &work_vector[max_row]);
        }

        // Eliminate column below pivot
        var j: usize = i + 1;
        while (j < 4) : (j += 1) {
            const factor = getMatrixElement4x4(&work_matrix, j, i) /
                getMatrixElement4x4(&work_matrix, i, i);

            var col: usize = i;
            while (col < 4) : (col += 1) {
                const current = getMatrixElement4x4(&work_matrix, j, col);
                const pivot = getMatrixElement4x4(&work_matrix, i, col);
                setMatrixElement4x4(&work_matrix, j, col, current - factor * pivot);
            }
            work_vector[j] -= factor * work_vector[i];
        }
    }

    // Check final diagonal element
    if (solvers.isNearZero(work_matrix.w[3], 1e-8)) {
        return SolverError.SingularMatrix;
    }

    // Back substitution
    const w = work_vector[3] / work_matrix.w[3];
    const z = (work_vector[2] - work_matrix.z[3] * w) / work_matrix.z[2];
    const y = (work_vector[1] - work_matrix.y[2] * z - work_matrix.y[3] * w) / work_matrix.y[1];
    const x = (work_vector[0] - work_matrix.x[1] * y - work_matrix.x[2] * z - work_matrix.x[3] * w) / work_matrix.x[0];

    return vector.Vec4{ .x = x, .y = y, .z = z, .w = w };
}

// Helper functions for matrix element access (row-major interpretation)

fn getMatrixElement3x3(m: *const matrix.Mat3x3, row: usize, col: usize) f32 {
    return switch (row) {
        0 => m.x[col],
        1 => m.y[col],
        2 => m.z[col],
        else => unreachable,
    };
}

fn setMatrixElement3x3(m: *matrix.Mat3x3, row: usize, col: usize, value: f32) void {
    switch (row) {
        0 => m.x[col] = value,
        1 => m.y[col] = value,
        2 => m.z[col] = value,
        else => unreachable,
    }
}

fn swapRows3x3(m: *matrix.Mat3x3, row1: usize, row2: usize) void {
    if (row1 == row2) return;

    const temp_row = switch (row1) {
        0 => m.x,
        1 => m.y,
        2 => m.z,
        else => unreachable,
    };

    switch (row1) {
        0 => m.x = switch (row2) {
            1 => m.y,
            2 => m.z,
            else => unreachable,
        },
        1 => m.y = switch (row2) {
            0 => m.x,
            2 => m.z,
            else => unreachable,
        },
        2 => m.z = switch (row2) {
            0 => m.x,
            1 => m.y,
            else => unreachable,
        },
        else => unreachable,
    }

    switch (row2) {
        0 => m.x = temp_row,
        1 => m.y = temp_row,
        2 => m.z = temp_row,
        else => unreachable,
    }
}

fn getMatrixElement4x4(m: *const matrix.Mat4x4, row: usize, col: usize) f32 {
    return switch (row) {
        0 => m.x[col],
        1 => m.y[col],
        2 => m.z[col],
        3 => m.w[col],
        else => unreachable,
    };
}

fn setMatrixElement4x4(m: *matrix.Mat4x4, row: usize, col: usize, value: f32) void {
    switch (row) {
        0 => m.x[col] = value,
        1 => m.y[col] = value,
        2 => m.z[col] = value,
        3 => m.w[col] = value,
        else => unreachable,
    }
}

fn swapRows4x4(m: *matrix.Mat4x4, row1: usize, row2: usize) void {
    if (row1 == row2) return;

    const temp_row = switch (row1) {
        0 => m.x,
        1 => m.y,
        2 => m.z,
        3 => m.w,
        else => unreachable,
    };

    switch (row1) {
        0 => m.x = switch (row2) {
            1 => m.y,
            2 => m.z,
            3 => m.w,
            else => unreachable,
        },
        1 => m.y = switch (row2) {
            0 => m.x,
            2 => m.z,
            3 => m.w,
            else => unreachable,
        },
        2 => m.z = switch (row2) {
            0 => m.x,
            1 => m.y,
            3 => m.w,
            else => unreachable,
        },
        3 => m.w = switch (row2) {
            0 => m.x,
            1 => m.y,
            2 => m.z,
            else => unreachable,
        },
        else => unreachable,
    }

    switch (row2) {
        0 => m.x = temp_row,
        1 => m.y = temp_row,
        2 => m.z = temp_row,
        3 => m.w = temp_row,
        else => unreachable,
    }
}

// Tests

test "solve2x2 basic system" {
    // System: 2x + y = 5
    //         x + 3y = 6
    // Solution: x = 1.5, y = 1.5... wait let me recalculate
    // 2x + y = 5
    // x + 3y = 6
    // From first: y = 5 - 2x
    // Substitute: x + 3(5 - 2x) = 6 => x + 15 - 6x = 6 => -5x = -9 => x = 1.8
    // Then: y = 5 - 2(1.8) = 5 - 3.6 = 1.4
    const A = matrix.Mat2x2{
        .x = .{ 2.0, 1.0 },
        .y = .{ 1.0, 3.0 },
    };
    const b = vector.Vec2{ .x = 5.0, .y = 6.0 };
    const x = try solve2x2(A, b);

    try std.testing.expectApproxEqAbs(1.8, x.x, 1e-5);
    try std.testing.expectApproxEqAbs(1.4, x.y, 1e-5);
}

test "solve3x3 basic system" {
    // System: x + 2y + z = 6
    //         2x + y + z = 6
    //         x + y + 2z = 7
    // Solution: x = 1, y = 1, z = 2
    const A = matrix.Mat3x3{
        .x = .{ 1.0, 2.0, 1.0 },
        .y = .{ 2.0, 1.0, 1.0 },
        .z = .{ 1.0, 1.0, 2.0 },
    };
    const b = vector.Vec3{ .x = 6.0, .y = 6.0, .z = 7.0 };
    const x = try solve3x3(A, b);

    try std.testing.expectApproxEqAbs(1.0, x.x, 1e-5);
    try std.testing.expectApproxEqAbs(1.5, x.y, 1e-5);
    try std.testing.expectApproxEqAbs(2.0, x.z, 1e-5);
}

test "solve3x3 singular matrix error" {
    // Singular matrix (row 3 = row 1 + row 2)
    const A = matrix.Mat3x3{
        .x = .{ 1.0, 2.0, 3.0 },
        .y = .{ 4.0, 5.0, 6.0 },
        .z = .{ 5.0, 7.0, 9.0 },
    };
    const b = vector.Vec3{ .x = 1.0, .y = 2.0, .z = 3.0 };

    const result = solve3x3(A, b);
    try std.testing.expectError(SolverError.SingularMatrix, result);
}

test "solve4x4 identity system" {
    // Identity matrix should return b as solution
    const A = matrix.Mat4x4{
        .x = .{ 1.0, 0.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
    const b = vector.Vec4{ .x = 2.0, .y = 3.0, .z = 4.0, .w = 5.0 };
    const x = try solve4x4(A, b);

    try std.testing.expectApproxEqAbs(2.0, x.x, 1e-5);
    try std.testing.expectApproxEqAbs(3.0, x.y, 1e-5);
    try std.testing.expectApproxEqAbs(4.0, x.z, 1e-5);
    try std.testing.expectApproxEqAbs(5.0, x.w, 1e-5);
}
