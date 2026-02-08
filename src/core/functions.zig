//! Linear algebra functions and transformations
//! This module provides essential linear algebra operations including
//! transformations, orthogonalization, projections, and utilities.

const std = @import("std");
const math = std.math;
const vector = @import("vector.zig");
const matrix = @import("matrix.zig");

const Vec2 = vector.Vec2;
const Vec3 = vector.Vec3;
const Vec4 = vector.Vec4;
const Mat2x2 = matrix.Mat2x2;
const Mat3x3 = matrix.Mat3x3;
const Mat4x4 = matrix.Mat4x4;

// ============================================================================
// Identity Matrices
// ============================================================================

/// Returns a 2x2 identity matrix
pub fn identity2x2() Mat2x2 {
    return .{
        .x = .{ 1.0, 0.0 },
        .y = .{ 0.0, 1.0 },
    };
}

/// Returns a 3x3 identity matrix
pub fn identity3x3() Mat3x3 {
    return .{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
}

/// Returns a 4x4 identity matrix
pub fn identity4x4() Mat4x4 {
    return .{
        .x = .{ 1.0, 0.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

// ============================================================================
// 2D Transformations
// ============================================================================

/// Creates a 2D rotation matrix (counter-clockwise)
/// angle: rotation angle in radians
pub fn rotation2D(angle: f32) Mat2x2 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ c, -s },
        .y = .{ s, c },
    };
}

/// Creates a 2D scaling matrix
pub fn scaling2D(sx: f32, sy: f32) Mat2x2 {
    return .{
        .x = .{ sx, 0.0 },
        .y = .{ 0.0, sy },
    };
}

/// Creates a 2D shearing matrix
/// shear_x: amount to shear in x direction
/// shear_y: amount to shear in y direction
pub fn shearing2D(shear_x: f32, shear_y: f32) Mat2x2 {
    return .{
        .x = .{ 1.0, shear_x },
        .y = .{ shear_y, 1.0 },
    };
}

/// Creates a 2D reflection matrix across the x-axis
pub fn reflectionX2D() Mat2x2 {
    return .{
        .x = .{ 1.0, 0.0 },
        .y = .{ 0.0, -1.0 },
    };
}

/// Creates a 2D reflection matrix across the y-axis
pub fn reflectionY2D() Mat2x2 {
    return .{
        .x = .{ -1.0, 0.0 },
        .y = .{ 0.0, 1.0 },
    };
}

/// Creates a 2D reflection matrix across a line through the origin at angle theta
pub fn reflectionLine2D(angle: f32) Mat2x2 {
    const c = @cos(2.0 * angle);
    const s = @sin(2.0 * angle);
    return .{
        .x = .{ c, s },
        .y = .{ s, -c },
    };
}

// ============================================================================
// 3D Transformations
// ============================================================================

/// Creates a 3D rotation matrix around the X-axis
pub fn rotationX(angle: f32) Mat3x3 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, c, -s },
        .z = .{ 0.0, s, c },
    };
}

/// Creates a 3D rotation matrix around the Y-axis
pub fn rotationY(angle: f32) Mat3x3 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ c, 0.0, s },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ -s, 0.0, c },
    };
}

/// Creates a 3D rotation matrix around the Z-axis
pub fn rotationZ(angle: f32) Mat3x3 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ c, -s, 0.0 },
        .y = .{ s, c, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
}

/// Creates a 3D rotation matrix around an arbitrary axis
/// axis: must be a normalized vector
/// angle: rotation angle in radians
pub fn rotationAxisAngle(axis: Vec3, angle: f32) Mat3x3 {
    const c = @cos(angle);
    const s = @sin(angle);
    const t = 1.0 - c;

    return .{
        .x = .{
            t * axis.x * axis.x + c,
            t * axis.x * axis.y - s * axis.z,
            t * axis.x * axis.z + s * axis.y,
        },
        .y = .{
            t * axis.x * axis.y + s * axis.z,
            t * axis.y * axis.y + c,
            t * axis.y * axis.z - s * axis.x,
        },
        .z = .{
            t * axis.x * axis.z - s * axis.y,
            t * axis.y * axis.z + s * axis.x,
            t * axis.z * axis.z + c,
        },
    };
}

/// Creates a 3D scaling matrix
pub fn scaling3D(sx: f32, sy: f32, sz: f32) Mat3x3 {
    return .{
        .x = .{ sx, 0.0, 0.0 },
        .y = .{ 0.0, sy, 0.0 },
        .z = .{ 0.0, 0.0, sz },
    };
}

/// Creates a 3D reflection matrix across the XY plane
pub fn reflectionXY() Mat3x3 {
    return .{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ 0.0, 0.0, -1.0 },
    };
}

/// Creates a 3D reflection matrix across the XZ plane
pub fn reflectionXZ() Mat3x3 {
    return .{
        .x = .{ 1.0, 0.0, 0.0 },
        .y = .{ 0.0, -1.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
}

/// Creates a 3D reflection matrix across the YZ plane
pub fn reflectionYZ() Mat3x3 {
    return .{
        .x = .{ -1.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0 },
    };
}

// ============================================================================
// 4x4 Homogeneous Transformations (for 3D graphics)
// ============================================================================

/// Creates a 4x4 translation matrix
pub fn translation(tx: f32, ty: f32, tz: f32) Mat4x4 {
    return .{
        .x = .{ 1.0, 0.0, 0.0, tx },
        .y = .{ 0.0, 1.0, 0.0, ty },
        .z = .{ 0.0, 0.0, 1.0, tz },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

/// Creates a 4x4 scaling matrix (homogeneous coordinates)
pub fn scaling4D(sx: f32, sy: f32, sz: f32) Mat4x4 {
    return .{
        .x = .{ sx, 0.0, 0.0, 0.0 },
        .y = .{ 0.0, sy, 0.0, 0.0 },
        .z = .{ 0.0, 0.0, sz, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

/// Creates a 4x4 rotation matrix around the X-axis
pub fn rotationX4x4(angle: f32) Mat4x4 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ 1.0, 0.0, 0.0, 0.0 },
        .y = .{ 0.0, c, -s, 0.0 },
        .z = .{ 0.0, s, c, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

/// Creates a 4x4 rotation matrix around the Y-axis
pub fn rotationY4x4(angle: f32) Mat4x4 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ c, 0.0, s, 0.0 },
        .y = .{ 0.0, 1.0, 0.0, 0.0 },
        .z = .{ -s, 0.0, c, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

/// Creates a 4x4 rotation matrix around the Z-axis
pub fn rotationZ4x4(angle: f32) Mat4x4 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{
        .x = .{ c, -s, 0.0, 0.0 },
        .y = .{ s, c, 0.0, 0.0 },
        .z = .{ 0.0, 0.0, 1.0, 0.0 },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

// ============================================================================
// Projection Matrices
// ============================================================================

/// Creates a perspective projection matrix (right-handed, OpenGL style)
/// fov_y: vertical field of view in radians
/// aspect: aspect ratio (width/height)
/// near: near clipping plane
/// far: far clipping plane
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4x4 {
    const tan_half_fov = @tan(fov_y / 2.0);
    return .{
        .x = .{ 1.0 / (aspect * tan_half_fov), 0.0, 0.0, 0.0 },
        .y = .{ 0.0, 1.0 / tan_half_fov, 0.0, 0.0 },
        .z = .{ 0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near) },
        .w = .{ 0.0, 0.0, -1.0, 0.0 },
    };
}

/// Creates an orthographic projection matrix (right-handed)
pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4x4 {
    return .{
        .x = .{ 2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left) },
        .y = .{ 0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom) },
        .z = .{ 0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near) },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

/// Creates a look-at view matrix (right-handed)
/// eye: camera position
/// target: point to look at
/// up: up direction (should be normalized)
pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4x4 {
    const f = target.sub(&eye).normalize();
    const s = f.cross(&up).normalize();
    const u = s.cross(&f);

    return .{
        .x = .{ s.x, s.y, s.z, -s.dot(&eye) },
        .y = .{ u.x, u.y, u.z, -u.dot(&eye) },
        .z = .{ -f.x, -f.y, -f.z, f.dot(&eye) },
        .w = .{ 0.0, 0.0, 0.0, 1.0 },
    };
}

// ============================================================================
// Matrix-Vector Operations
// ============================================================================

/// Transforms a 2D vector by a 2x2 matrix
pub fn transform2D(m: *const Mat2x2, v: *const Vec2) Vec2 {
    return .{
        .x = m.x[0] * v.x + m.x[1] * v.y,
        .y = m.y[0] * v.x + m.y[1] * v.y,
    };
}

/// Transforms a 3D vector by a 3x3 matrix
pub fn transform3D(m: *const Mat3x3, v: *const Vec3) Vec3 {
    return .{
        .x = m.x[0] * v.x + m.x[1] * v.y + m.x[2] * v.z,
        .y = m.y[0] * v.x + m.y[1] * v.y + m.y[2] * v.z,
        .z = m.z[0] * v.x + m.z[1] * v.y + m.z[2] * v.z,
    };
}

/// Transforms a 4D vector by a 4x4 matrix
pub fn transform4D(m: *const Mat4x4, v: *const Vec4) Vec4 {
    return .{
        .x = m.x[0] * v.x + m.x[1] * v.y + m.x[2] * v.z + m.x[3] * v.w,
        .y = m.y[0] * v.x + m.y[1] * v.y + m.y[2] * v.z + m.y[3] * v.w,
        .z = m.z[0] * v.x + m.z[1] * v.y + m.z[2] * v.z + m.z[3] * v.w,
        .w = m.w[0] * v.x + m.w[1] * v.y + m.w[2] * v.z + m.w[3] * v.w,
    };
}

/// Transforms a 3D point by a 4x4 matrix (assumes w=1 for point)
pub fn transformPoint(m: *const Mat4x4, v: *const Vec3) Vec3 {
    const w = m.w[0] * v.x + m.w[1] * v.y + m.w[2] * v.z + m.w[3];
    return .{
        .x = (m.x[0] * v.x + m.x[1] * v.y + m.x[2] * v.z + m.x[3]) / w,
        .y = (m.y[0] * v.x + m.y[1] * v.y + m.y[2] * v.z + m.y[3]) / w,
        .z = (m.z[0] * v.x + m.z[1] * v.y + m.z[2] * v.z + m.z[3]) / w,
    };
}

/// Transforms a 3D direction by a 4x4 matrix (assumes w=0 for direction)
pub fn transformDirection(m: *const Mat4x4, v: *const Vec3) Vec3 {
    return .{
        .x = m.x[0] * v.x + m.x[1] * v.y + m.x[2] * v.z,
        .y = m.y[0] * v.x + m.y[1] * v.y + m.y[2] * v.z,
        .z = m.z[0] * v.x + m.z[1] * v.y + m.z[2] * v.z,
    };
}

// ============================================================================
// Orthogonal Operations
// ============================================================================

/// Checks if two 2D vectors are orthogonal (perpendicular)
/// epsilon: tolerance for floating-point comparison
pub fn isOrthogonal2D(a: *const Vec2, b: *const Vec2, epsilon: f32) bool {
    return @abs(a.dot(b)) < epsilon;
}

/// Checks if two 3D vectors are orthogonal (perpendicular)
/// epsilon: tolerance for floating-point comparison
pub fn isOrthogonal3D(a: *const Vec3, b: *const Vec3, epsilon: f32) bool {
    return @abs(a.dot(b)) < epsilon;
}

/// Checks if a 2x2 matrix is orthogonal (preserves lengths and angles)
/// An orthogonal matrix satisfies: M^T * M = I
pub fn isOrthogonalMatrix2x2(m: *const Mat2x2, epsilon: f32) bool {
    const mt = m.transpose();
    const product = mt.mul(m);
    const identity = identity2x2();

    return @abs(product.x[0] - identity.x[0]) < epsilon and
        @abs(product.x[1] - identity.x[1]) < epsilon and
        @abs(product.y[0] - identity.y[0]) < epsilon and
        @abs(product.y[1] - identity.y[1]) < epsilon;
}

/// Checks if a 3x3 matrix is orthogonal
pub fn isOrthogonalMatrix3x3(m: *const Mat3x3, epsilon: f32) bool {
    const mt = m.transpose();
    const product = mt.mul(m);
    const identity = identity3x3();

    return @abs(product.x[0] - identity.x[0]) < epsilon and
        @abs(product.x[1] - identity.x[1]) < epsilon and
        @abs(product.x[2] - identity.x[2]) < epsilon and
        @abs(product.y[0] - identity.y[0]) < epsilon and
        @abs(product.y[1] - identity.y[1]) < epsilon and
        @abs(product.y[2] - identity.y[2]) < epsilon and
        @abs(product.z[0] - identity.z[0]) < epsilon and
        @abs(product.z[1] - identity.z[1]) < epsilon and
        @abs(product.z[2] - identity.z[2]) < epsilon;
}

/// Projects vector a onto vector b (orthogonal projection)
pub fn projectVec2(a: *const Vec2, b: *const Vec2) Vec2 {
    const dot_ab = a.dot(b);
    const dot_bb = b.dot(b);
    if (dot_bb == 0) {
        return .{ .x = 0, .y = 0 };
    }
    const scalar = dot_ab / dot_bb;
    return b.scale(scalar);
}

/// Projects vector a onto vector b (orthogonal projection)
pub fn projectVec3(a: *const Vec3, b: *const Vec3) Vec3 {
    const dot_ab = a.dot(b);
    const dot_bb = b.dot(b);
    if (dot_bb == 0) {
        return .{ .x = 0, .y = 0, .z = 0 };
    }
    const scalar = dot_ab / dot_bb;
    return b.scale(scalar);
}

/// Gram-Schmidt orthogonalization for 2 vectors in 2D
/// Returns two orthonormal vectors
pub fn gramSchmidt2D(v1: Vec2, v2: Vec2) [2]Vec2 {
    // Normalize v1
    const basis1 = v1.normalize();

    // Subtract projection of v2 onto basis1
    const proj = projectVec2(&v2, &basis1);
    const u2_unnormalized = v2.sub(&proj);
    const basis2 = u2_unnormalized.normalize();

    return .{ basis1, basis2 };
}

/// Gram-Schmidt orthogonalization for 3 vectors in 3D
/// Returns three orthonormal vectors
pub fn gramSchmidt3D(v1: Vec3, v2: Vec3, v3: Vec3) [3]Vec3 {
    // Normalize v1
    const basis1 = v1.normalize();

    // Orthogonalize v2
    const proj1 = projectVec3(&v2, &basis1);
    const u2_unnormalized = v2.sub(&proj1);
    const basis2 = u2_unnormalized.normalize();

    // Orthogonalize v3
    const proj2 = projectVec3(&v3, &basis1);
    const proj3 = projectVec3(&v3, &basis2);
    const sum_proj = proj2.add(&proj3);
    const u3_unnormalized = v3.sub(&sum_proj);
    const basis3 = u3_unnormalized.normalize();

    return .{ basis1, basis2, basis3 };
}

// ============================================================================
// Matrix Utilities
// ============================================================================

/// Calculates the trace of a 2x2 matrix (sum of diagonal elements)
pub fn trace2x2(m: *const Mat2x2) f32 {
    return m.x[0] + m.y[1];
}

/// Calculates the trace of a 3x3 matrix
pub fn trace3x3(m: *const Mat3x3) f32 {
    return m.x[0] + m.y[1] + m.z[2];
}

/// Calculates the trace of a 4x4 matrix
pub fn trace4x4(m: *const Mat4x4) f32 {
    return m.x[0] + m.y[1] + m.z[2] + m.w[3];
}

/// Calculates the Frobenius norm of a 2x2 matrix
pub fn frobeniusNorm2x2(m: *const Mat2x2) f32 {
    return @sqrt(
        m.x[0] * m.x[0] + m.x[1] * m.x[1] +
            m.y[0] * m.y[0] + m.y[1] * m.y[1],
    );
}

/// Calculates the Frobenius norm of a 3x3 matrix
pub fn frobeniusNorm3x3(m: *const Mat3x3) f32 {
    return @sqrt(
        m.x[0] * m.x[0] + m.x[1] * m.x[1] + m.x[2] * m.x[2] +
            m.y[0] * m.y[0] + m.y[1] * m.y[1] + m.y[2] * m.y[2] +
            m.z[0] * m.z[0] + m.z[1] * m.z[1] + m.z[2] * m.z[2],
    );
}

/// Calculates the Frobenius norm of a 4x4 matrix
pub fn frobeniusNorm4x4(m: *const Mat4x4) f32 {
    return @sqrt(
        m.x[0] * m.x[0] + m.x[1] * m.x[1] + m.x[2] * m.x[2] + m.x[3] * m.x[3] +
            m.y[0] * m.y[0] + m.y[1] * m.y[1] + m.y[2] * m.y[2] + m.y[3] * m.y[3] +
            m.z[0] * m.z[0] + m.z[1] * m.z[1] + m.z[2] * m.z[2] + m.z[3] * m.z[3] +
            m.w[0] * m.w[0] + m.w[1] * m.w[1] + m.w[2] * m.w[2] + m.w[3] * m.w[3],
    );
}

/// Computes matrix power for 2x2 matrix (positive integer exponent only)
pub fn matrixPower2x2(m: *const Mat2x2, n: u32) Mat2x2 {
    if (n == 0) {
        return identity2x2();
    }
    if (n == 1) {
        return m.*;
    }

    var result = identity2x2();
    var base = m.*;
    var exp = n;

    // Fast exponentiation by squaring
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result.mul(&base);
        }
        base = base.mul(&base);
        exp /= 2;
    }

    return result;
}

/// Computes matrix power for 3x3 matrix (positive integer exponent only)
pub fn matrixPower3x3(m: *const Mat3x3, n: u32) Mat3x3 {
    if (n == 0) {
        return identity3x3();
    }
    if (n == 1) {
        return m.*;
    }

    var result = identity3x3();
    var base = m.*;
    var exp = n;

    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result.mul(&base);
        }
        base = base.mul(&base);
        exp /= 2;
    }

    return result;
}

/// Computes matrix power for 4x4 matrix (positive integer exponent only)
pub fn matrixPower4x4(m: *const Mat4x4, n: u32) Mat4x4 {
    if (n == 0) {
        return identity4x4();
    }
    if (n == 1) {
        return m.*;
    }

    var result = identity4x4();
    var base = m.*;
    var exp = n;

    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result.mul(&base);
        }
        base = base.mul(&base);
        exp /= 2;
    }

    return result;
}

// ============================================================================
// Angle Utilities
// ============================================================================

/// Computes the angle between two 2D vectors (in radians)
pub fn angleBetween2D(a: *const Vec2, b: *const Vec2) f32 {
    const dot = a.dot(b);
    const len_a = a.length();
    const len_b = b.length();
    if (len_a == 0 or len_b == 0) {
        return 0;
    }
    const cos_angle = dot / (len_a * len_b);
    return math.acos(std.math.clamp(cos_angle, -1.0, 1.0));
}

/// Computes the angle between two 3D vectors (in radians)
pub fn angleBetween3D(a: *const Vec3, b: *const Vec3) f32 {
    const dot = a.dot(b);
    const len_a = a.length();
    const len_b = b.length();
    if (len_a == 0 or len_b == 0) {
        return 0;
    }
    const cos_angle = dot / (len_a * len_b);
    return math.acos(std.math.clamp(cos_angle, -1.0, 1.0));
}

// ============================================================================
// Linear Interpolation
// ============================================================================

/// Linear interpolation between two 2D vectors
/// t should be in range [0, 1]
pub fn lerp2D(a: *const Vec2, b: *const Vec2, t: f32) Vec2 {
    return .{
        .x = a.x + (b.x - a.x) * t,
        .y = a.y + (b.y - a.y) * t,
    };
}

/// Linear interpolation between two 3D vectors
/// t should be in range [0, 1]
pub fn lerp3D(a: *const Vec3, b: *const Vec3, t: f32) Vec3 {
    return .{
        .x = a.x + (b.x - a.x) * t,
        .y = a.y + (b.y - a.y) * t,
        .z = a.z + (b.z - a.z) * t,
    };
}

/// Linear interpolation between two 4D vectors
/// t should be in range [0, 1]
pub fn lerp4D(a: *const Vec4, b: *const Vec4, t: f32) Vec4 {
    return .{
        .x = a.x + (b.x - a.x) * t,
        .y = a.y + (b.y - a.y) * t,
        .z = a.z + (b.z - a.z) * t,
        .w = a.w + (b.w - a.w) * t,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "identity matrices" {
    const id2 = identity2x2();
    try testing.expectEqual(@as(f32, 1.0), id2.x[0]);
    try testing.expectEqual(@as(f32, 0.0), id2.x[1]);

    const id3 = identity3x3();
    try testing.expectEqual(@as(f32, 1.0), id3.y[1]);

    const id4 = identity4x4();
    try testing.expectEqual(@as(f32, 1.0), id4.w[3]);
}

test "2D rotation" {
    const rot = rotation2D(math.pi / 2.0); // 90 degrees
    const v = Vec2{ .x = 1.0, .y = 0.0 };
    const result = transform2D(&rot, &v);

    try testing.expectApproxEqAbs(@as(f32, 0.0), result.x, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result.y, 0.0001);
}

test "3D rotation X" {
    const rot = rotationX(math.pi / 2.0);
    const v = Vec3{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const result = transform3D(&rot, &v);

    try testing.expectApproxEqAbs(@as(f32, 0.0), result.x, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result.y, 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result.z, 0.0001);
}

test "scaling" {
    const scale = scaling2D(2.0, 3.0);
    const v = Vec2{ .x = 1.0, .y = 1.0 };
    const result = transform2D(&scale, &v);

    try testing.expectEqual(@as(f32, 2.0), result.x);
    try testing.expectEqual(@as(f32, 3.0), result.y);
}

test "orthogonality check" {
    const v1 = Vec2{ .x = 1.0, .y = 0.0 };
    const v2 = Vec2{ .x = 0.0, .y = 1.0 };

    try testing.expect(isOrthogonal2D(&v1, &v2, 0.0001));
}

test "projection" {
    const a = Vec3{ .x = 1.0, .y = 1.0, .z = 0.0 };
    const b = Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const proj = projectVec3(&a, &b);

    try testing.expectEqual(@as(f32, 1.0), proj.x);
    try testing.expectEqual(@as(f32, 0.0), proj.y);
    try testing.expectEqual(@as(f32, 0.0), proj.z);
}

test "Gram-Schmidt 2D" {
    const v1 = Vec2{ .x = 1.0, .y = 0.0 };
    const v2 = Vec2{ .x = 1.0, .y = 1.0 };
    const result = gramSchmidt2D(v1, v2);

    // Result should be orthonormal
    try testing.expect(isOrthogonal2D(&result[0], &result[1], 0.0001));
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0].length(), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[1].length(), 0.0001);
}

test "trace" {
    const m = Mat3x3{
        .x = .{ 1.0, 2.0, 3.0 },
        .y = .{ 4.0, 5.0, 6.0 },
        .z = .{ 7.0, 8.0, 9.0 },
    };
    const tr = trace3x3(&m);
    try testing.expectEqual(@as(f32, 15.0), tr);
}

test "matrix power" {
    const m = scaling2D(2.0, 2.0);
    const m_squared = matrixPower2x2(&m, 2);

    try testing.expectEqual(@as(f32, 4.0), m_squared.x[0]);
    try testing.expectEqual(@as(f32, 4.0), m_squared.y[1]);
}

test "angle between vectors" {
    const v1 = Vec2{ .x = 1.0, .y = 0.0 };
    const v2 = Vec2{ .x = 0.0, .y = 1.0 };
    const angle = angleBetween2D(&v1, &v2);

    try testing.expectApproxEqAbs(math.pi / 2.0, angle, 0.0001);
}

test "lerp" {
    const a = Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const b = Vec3{ .x = 10.0, .y = 10.0, .z = 10.0 };
    const mid = lerp3D(&a, &b, 0.5);

    try testing.expectEqual(@as(f32, 5.0), mid.x);
    try testing.expectEqual(@as(f32, 5.0), mid.y);
    try testing.expectEqual(@as(f32, 5.0), mid.z);
}
