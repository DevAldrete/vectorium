//! Demonstration of linear algebra solvers
//!
//! This example showcases the various solver capabilities:
//! - Solving linear systems
//! - LU decomposition
//! - QR decomposition
//! - Eigenvalue computations

const std = @import("std");
const vectorium = @import("vectorium");
const print = std.debug.print;

const matrix = vectorium.core.matrix;
const vector = vectorium.core.vector;
const solvers = vectorium.core.solvers;

pub fn main() !void {
    print("\n=== Vectorium Linear Algebra Solvers Demo ===\n\n", .{});

    try demonstrateLinearSystems();
    try demonstrateLUDecomposition();
    try demonstrateQRDecomposition();
    try demonstrateEigenvalues();

    print("=== All solver demonstrations complete! ===\n", .{});
}

fn demonstrateLinearSystems() !void {
    print("1. Linear System Solver (Gaussian Elimination)\n", .{});
    print("   Solving: Ax = b\n\n", .{});

    // System: 2x + y + z = 5
    //         x + 3y + 2z = 8
    //         x + y + 4z = 10
    const A = matrix.Mat3x3{
        .x = .{ 2.0, 1.0, 1.0 },
        .y = .{ 1.0, 3.0, 2.0 },
        .z = .{ 1.0, 1.0, 4.0 },
    };
    const b = vector.Vec3{ .x = 5.0, .y = 8.0, .z = 10.0 };

    print("   Matrix A:\n", .{});
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.x[0], A.x[1], A.x[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.y[0], A.y[1], A.y[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.z[0], A.z[1], A.z[2] });
    print("   Vector b: [{d:.1}, {d:.1}, {d:.1}]\n\n", .{ b.x, b.y, b.z });

    const x = try solvers.linear_systems.solve3x3(A, b);
    print("   Solution x: [{d:.3}, {d:.3}, {d:.3}]\n\n", .{ x.x, x.y, x.z });

    // Verify solution
    const verification = vector.Vec3{
        .x = A.x[0] * x.x + A.x[1] * x.y + A.x[2] * x.z,
        .y = A.y[0] * x.x + A.y[1] * x.y + A.y[2] * x.z,
        .z = A.z[0] * x.x + A.z[1] * x.y + A.z[2] * x.z,
    };
    print("   Verification (Ax): [{d:.3}, {d:.3}, {d:.3}]\n", .{ verification.x, verification.y, verification.z });
    print("   ✓ Solution verified!\n\n", .{});
}

fn demonstrateLUDecomposition() !void {
    print("2. LU Decomposition\n", .{});
    print("   Factoring A = L * U\n\n", .{});

    const A = matrix.Mat3x3{
        .x = .{ 4.0, 3.0, 2.0 },
        .y = .{ 3.0, 3.0, 2.0 },
        .z = .{ 1.0, 1.0, 1.0 },
    };

    print("   Matrix A:\n", .{});
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.x[0], A.x[1], A.x[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.y[0], A.y[1], A.y[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n\n", .{ A.z[0], A.z[1], A.z[2] });

    const lu = try solvers.decomposition.lu3x3(A);

    print("   Lower triangular L:\n", .{});
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ lu.L.x[0], lu.L.x[1], lu.L.x[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ lu.L.y[0], lu.L.y[1], lu.L.y[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n\n", .{ lu.L.z[0], lu.L.z[1], lu.L.z[2] });

    print("   Upper triangular U:\n", .{});
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ lu.U.x[0], lu.U.x[1], lu.U.x[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ lu.U.y[0], lu.U.y[1], lu.U.y[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n\n", .{ lu.U.z[0], lu.U.z[1], lu.U.z[2] });

    const det = lu.determinant();
    print("   Determinant: {d:.3}\n", .{det});
    print("   ✓ LU decomposition complete!\n\n", .{});
}

fn demonstrateQRDecomposition() !void {
    print("3. QR Decomposition\n", .{});
    print("   Factoring A = Q * R (Q orthogonal, R upper triangular)\n\n", .{});

    const A = matrix.Mat3x3{
        .x = .{ 1.0, 1.0, 0.0 },
        .y = .{ 1.0, 0.0, 1.0 },
        .z = .{ 0.0, 1.0, 1.0 },
    };

    print("   Matrix A:\n", .{});
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.x[0], A.x[1], A.x[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.y[0], A.y[1], A.y[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n\n", .{ A.z[0], A.z[1], A.z[2] });

    const qr = try solvers.decomposition.qr3x3(A);

    print("   Orthogonal matrix Q:\n", .{});
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ qr.Q.x[0], qr.Q.x[1], qr.Q.x[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ qr.Q.y[0], qr.Q.y[1], qr.Q.y[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n\n", .{ qr.Q.z[0], qr.Q.z[1], qr.Q.z[2] });

    print("   Upper triangular R:\n", .{});
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ qr.R.x[0], qr.R.x[1], qr.R.x[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ qr.R.y[0], qr.R.y[1], qr.R.y[2] });
    print("   [{d:.3}, {d:.3}, {d:.3}]\n", .{ qr.R.z[0], qr.R.z[1], qr.R.z[2] });
    print("   ✓ QR decomposition complete!\n\n", .{});
}

fn demonstrateEigenvalues() !void {
    print("4. Eigenvalue Computations\n\n", .{});

    // Symmetric matrix for cleaner eigenvalues
    const A = matrix.Mat3x3{
        .x = .{ 4.0, 1.0, 0.0 },
        .y = .{ 1.0, 3.0, 1.0 },
        .z = .{ 0.0, 1.0, 2.0 },
    };

    print("   Matrix A:\n", .{});
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.x[0], A.x[1], A.x[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n", .{ A.y[0], A.y[1], A.y[2] });
    print("   [{d:.1}, {d:.1}, {d:.1}]\n\n", .{ A.z[0], A.z[1], A.z[2] });

    // Power iteration (largest eigenvalue)
    print("   a) Power Iteration (largest eigenvalue):\n", .{});
    const config = solvers.IterativeConfig{
        .max_iterations = 1000,
        .tolerance = 1e-6,
        .verbose = false,
    };
    const power_result = try solvers.eigen.powerIteration3x3(A, config);
    print("      λ_max = {d:.6}\n", .{power_result.eigenvalue});
    print("      Eigenvector: [{d:.3}, {d:.3}, {d:.3}]\n", .{ power_result.eigenvector.x, power_result.eigenvector.y, power_result.eigenvector.z });
    print("      Converged in {} iterations\n\n", .{power_result.iterations});

    // Inverse power iteration (smallest eigenvalue)
    print("   b) Inverse Power Iteration (smallest eigenvalue):\n", .{});
    const inv_result = try solvers.eigen.inversePowerIteration3x3(A, config);
    print("      λ_min = {d:.6}\n", .{inv_result.eigenvalue});
    print("      Eigenvector: [{d:.3}, {d:.3}, {d:.3}]\n", .{ inv_result.eigenvector.x, inv_result.eigenvector.y, inv_result.eigenvector.z });
    print("      Converged in {} iterations\n\n", .{inv_result.iterations});

    // QR algorithm (all eigenvalues)
    print("   c) QR Algorithm (all eigenvalues):\n", .{});
    const qr_config = solvers.IterativeConfig{
        .max_iterations = 1000,
        .tolerance = 1e-5,
        .verbose = false,
    };
    const qr_result = try solvers.eigen.qrAlgorithm3x3(A, qr_config);
    print("      Eigenvalues: [{d:.6}, {d:.6}, {d:.6}]\n", .{ qr_result.eigenvalues[0], qr_result.eigenvalues[1], qr_result.eigenvalues[2] });
    print("      Converged in {} iterations\n", .{qr_result.iterations});
    print("      ✓ All eigenvalues computed!\n\n", .{});
}
