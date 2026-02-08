const std = @import("std");
const math = std.math;
const vector = @import("vector.zig");

// 2x2 Matrix struct with the essential operations
pub const Mat2x2 = struct {
    const Self = @This();

    x: [2]f32,
    y: [2]f32,

    pub fn add(self: *const Self, b: *const Mat2x2) Mat2x2 {
        return .{
            .x = .{ self.x[0] + b.x[0], self.x[1] + b.x[1] },
            .y = .{ self.y[0] + b.y[0], self.y[1] + b.y[1] },
        };
    }

    pub fn sub(self: *const Self, b: *const Mat2x2) Mat2x2 {
        return .{
            .x = .{ self.x[0] - b.x[0], self.x[1] - b.x[1] },
            .y = .{ self.y[0] - b.y[0], self.y[1] - b.y[1] },
        };
    }

    pub fn mul(self: *const Self, b: *const Mat2x2) Mat2x2 {
        return .{
            .x = .{
                self.x[0] * b.x[0] + self.x[1] * b.y[0],
                self.x[0] * b.x[1] + self.x[1] * b.y[1],
            },
            .y = .{
                self.y[0] * b.x[0] + self.y[1] * b.y[0],
                self.y[0] * b.x[1] + self.y[1] * b.y[1],
            },
        };
    }

    pub fn scale(self: *const Self, scalar: f32) Mat2x2 {
        return .{
            .x = .{ self.x[0] * scalar, self.x[1] * scalar },
            .y = .{ self.y[0] * scalar, self.y[1] * scalar },
        };
    }

    pub fn determinant(self: *const Self) f32 {
        return self.x[0] * self.y[1] - self.x[1] * self.y[0];
    }

    pub fn inverse(self: *const Self) ?Mat2x2 {
        const det = self.determinant();
        if (det == 0) {
            return null; // Not invertible
        }
        const inv_det = 1.0 / det;
        return .{
            .x = .{ self.y[1] * inv_det, -self.x[1] * inv_det },
            .y = .{ -self.y[0] * inv_det, self.x[0] * inv_det },
        };
    }

    pub fn transpose(self: *const Self) Mat2x2 {
        return .{
            .x = .{ self.x[0], self.y[0] },
            .y = .{ self.x[1], self.y[1] },
        };
    }
};

// 3x3 Matrix struct with the essential operations
pub const Mat3x3 = struct {
    const Self = @This();

    x: [3]f32,
    y: [3]f32,
    z: [3]f32,

    pub fn add(self: *const Self, b: *const Mat3x3) Mat3x3 {
        return .{
            .x = .{ self.x[0] + b.x[0], self.x[1] + b.x[1], self.x[2] + b.x[2] },
            .y = .{ self.y[0] + b.y[0], self.y[1] + b.y[1], self.y[2] + b.y[2] },
            .z = .{ self.z[0] + b.z[0], self.z[1] + b.z[1], self.z[2] + b.z[2] },
        };
    }

    pub fn sub(self: *const Self, b: *const Mat3x3) Mat3x3 {
        return .{
            .x = .{ self.x[0] - b.x[0], self.x[1] - b.x[1], self.x[2] - b.x[2] },
            .y = .{ self.y[0] - b.y[0], self.y[1] - b.y[1], self.y[2] - b.y[2] },
            .z = .{ self.z[0] - b.z[0], self.z[1] - b.z[1], self.z[2] - b.z[2] },
        };
    }

    pub fn mul(self: *const Self, b: *const Mat3x3) Mat3x3 {
        return .{
            .x = .{
                self.x[0] * b.x[0] + self.x[1] * b.y[0] + self.x[2] * b.z[0],
                self.x[0] * b.x[1] + self.x[1] * b.y[1] + self.x[2] * b.z[1],
                self.x[0] * b.x[2] + self.x[1] * b.y[2] + self.x[2] * b.z[2],
            },
            .y = .{
                self.y[0] * b.x[0] + self.y[1] * b.y[0] + self.y[2] * b.z[0],
                self.y[0] * b.x[1] + self.y[1] * b.y[1] + self.y[2] * b.z[1],
                self.y[0] * b.x[2] + self.y[1] * b.y[2] + self.y[2] * b.z[2],
            },
            .z = .{
                self.z[0] * b.x[0] + self.z[1] * b.y[0] + self.z[2] * b.z[0],
                self.z[0] * b.x[1] + self.z[1] * b.y[1] + self.z[2] * b.z[1],
                self.z[0] * b.x[2] + self.z[1] * b.y[2] + self.z[2] * b.z[2],
            },
        };
    }

    pub fn scale(self: *const Self, scalar: f32) Mat3x3 {
        return .{
            .x = .{ self.x[0] * scalar, self.x[1] * scalar, self.x[2] * scalar },
            .y = .{ self.y[0] * scalar, self.y[1] * scalar, self.y[2] * scalar },
            .z = .{ self.z[0] * scalar, self.z[1] * scalar, self.z[2] * scalar },
        };
    }

    pub fn determinant(self: *const Self) f32 {
        return self.x[0] * (self.y[1] * self.z[2] - self.y[2] * self.z[1]) -
            self.x[1] * (self.y[0] * self.z[2] - self.y[2] * self.z[0]) +
            self.x[2] * (self.y[0] * self.z[1] - self.y[1] * self.z[0]);
    }

    pub fn inverse(self: *const Self) ?Mat3x3 {
        const det = self.determinant();
        if (det == 0) {
            return null; // Not invertible
        }
        const inv_det = 1.0 / det;

        return .{
            .x = .{
                (self.y[1] * self.z[2] - self.y[2] * self.z[1]) * inv_det,
                (self.x[2] * self.z[1] - self.x[1] * self.z[2]) * inv_det,
                (self.x[1] * self.y[2] - self.x[2] * self.y[1]) * inv_det,
            },
            .y = .{
                (self.y[2] * self.z[0] - self.y[0] * self.z[2]) * inv_det,
                (self.x[0] * self.z[2] - self.x[2] * self.z[0]) * inv_det,
                (self.x[2] * self.y[0] - self.x[0] * self.y[2]) * inv_det,
            },
            .z = .{
                (self.y[0] * self.z[1] - self.y[1] * self.z[0]) * inv_det,
                (self.x[1] * self.z[0] - self.x[0] * self.z[1]) * inv_det,
                (self.x[0] * self.y[1] - self.x[1] * self.y[0]) * inv_det,
            },
        };
    }

    pub fn transpose(self: *const Self) Mat3x3 {
        return .{
            .x = .{ self.x[0], self.y[0], self.z[0] },
            .y = .{ self.x[1], self.y[1], self.z[1] },
            .z = .{ self.x[2], self.y[2], self.z[2] },
        };
    }
};

// Matrix4x4 struct with the essential operations (for 3D transformations)
pub const Mat4x4 = struct {
    const Self = @This();

    x: [4]f32,
    y: [4]f32,
    z: [4]f32,
    w: [4]f32,

    pub fn add(self: *const Self, b: *const Mat4x4) Mat4x4 {
        return .{
            .x = .{ self.x[0] + b.x[0], self.x[1] + b.x[1], self.x[2] + b.x[2], self.x[3] + b.x[3] },
            .y = .{ self.y[0] + b.y[0], self.y[1] + b.y[1], self.y[2] + b.y[2], self.y[3] + b.y[3] },
            .z = .{ self.z[0] + b.z[0], self.z[1] + b.z[1], self.z[2] + b.z[2], self.z[3] + b.z[3] },
            .w = .{ self.w[0] + b.w[0], self.w[1] + b.w[1], self.w[2] + b.w[2], self.w[3] + b.w[3] },
        };
    }

    pub fn sub(self: *const Self, b: *const Mat4x4) Mat4x4 {
        return .{
            .x = .{ self.x[0] - b.x[0], self.x[1] - b.x[1], self.x[2] - b.x[2], self.x[3] - b.x[3] },
            .y = .{ self.y[0] - b.y[0], self.y[1] - b.y[1], self.y[2] - b.y[2], self.y[3] - b.y[3] },
            .z = .{ self.z[0] - b.z[0], self.z[1] - b.z[1], self.z[2] - b.z[2], self.z[3] - b.z[3] },
            .w = .{ self.w[0] - b.w[0], self.w[1] - b.w[1], self.w[2] - b.w[2], self.w[3] - b.w[3] },
        };
    }

    pub fn mul(self: *const Self, b: *const Mat4x4) Mat4x4 {
        return .{
            .x = .{
                self.x[0] * b.x[0] + self.x[1] * b.y[0] + self.x[2] * b.z[0] + self.x[3] * b.w[0],
                self.x[0] * b.x[1] + self.x[1] * b.y[1] + self.x[2] * b.z[1] + self.x[3] * b.w[1],
                self.x[0] * b.x[2] + self.x[1] * b.y[2] + self.x[2] * b.z[2] + self.x[3] * b.w[2],
                self.x[0] * b.x[3] + self.x[1] * b.y[3] + self.x[2] * b.z[3] + self.x[3] * b.w[3],
            },
            .y = .{
                self.y[0] * b.x[0] + self.y[1] * b.y[0] + self.y[2] * b.z[0] + self.y[3] * b.w[0],
                self.y[0] * b.x[1] + self.y[1] * b.y[1] + self.y[2] * b.z[1] + self.y[3] * b.w[1],
                self.y[0] * b.x[2] + self.y[1] * b.y[2] + self.y[2] * b.z[2] + self.y[3] * b.w[2],
                self.y[0] * b.x[3] + self.y[1] * b.y[3] + self.y[2] * b.z[3] + self.y[3] * b.w[3],
            },
            .z = .{
                self.z[0] * b.x[0] + self.z[1] * b.y[0] + self.z[2] * b.z[0] + self.z[3] * b.w[0],
                self.z[0] * b.x[1] + self.z[1] * b.y[1] + self.z[2] * b.z[1] + self.z[3] * b.w[1],
                self.z[0] * b.x[2] + self.z[1] * b.y[2] + self.z[2] * b.z[2] + self.z[3] * b.w[2],
                self.z[0] * b.x[3] + self.z[1] * b.y[3] + self.z[2] * b.z[3] + self.z[3] * b.w[3],
            },
            .w = .{
                self.w[0] * b.x[0] + self.w[1] * b.y[0] + self.w[2] * b.z[0] + self.w[3] * b.w[0],
                self.w[0] * b.x[1] + self.w[1] * b.y[1] + self.w[2] * b.z[1] + self.w[3] * b.w[1],
                self.w[0] * b.x[2] + self.w[1] * b.y[2] + self.w[2] * b.z[2] + self.w[3] * b.w[2],
                self.w[0] * b.x[3] + self.w[1] * b.y[3] + self.w[2] * b.z[3] + self.w[3] * b.w[3],
            },
        };
    }

    pub fn scale(self: *const Self, scalar: f32) Mat4x4 {
        return .{
            .x = .{ self.x[0] * scalar, self.x[1] * scalar, self.x[2] * scalar, self.x[3] * scalar },
            .y = .{ self.y[0] * scalar, self.y[1] * scalar, self.y[2] * scalar, self.y[3] * scalar },
            .z = .{ self.z[0] * scalar, self.z[1] * scalar, self.z[2] * scalar, self.z[3] * scalar },
            .w = .{ self.w[0] * scalar, self.w[1] * scalar, self.w[2] * scalar, self.w[3] * scalar },
        };
    }

    pub fn determinant(self: *const Self) f32 {
        const a0 = self.x[0] * self.y[1] - self.x[1] * self.y[0];
        const a1 = self.x[0] * self.y[2] - self.x[2] * self.y[0];
        const a2 = self.x[0] * self.y[3] - self.x[3] * self.y[0];
        const a3 = self.x[1] * self.y[2] - self.x[2] * self.y[1];
        const a4 = self.x[1] * self.y[3] - self.x[3] * self.y[1];
        const a5 = self.x[2] * self.y[3] - self.x[3] * self.y[2];
        const b0 = self.z[0] * self.w[1] - self.z[1] * self.w[0];
        const b1 = self.z[0] * self.w[2] - self.z[2] * self.w[0];
        const b2 = self.z[0] * self.w[3] - self.z[3] * self.w[0];
        const b3 = self.z[1] * self.w[2] - self.z[2] * self.w[1];
        const b4 = self.z[1] * self.w[3] - self.z[3] * self.w[1];
        const b5 = self.z[2] * self.w[3] - self.z[3] * self.w[2];

        return a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;
    }

    pub fn inverse(self: *const Self) ?Mat4x4 {
        const a0 = self.x[0] * self.y[1] - self.x[1] * self.y[0];
        const a1 = self.x[0] * self.y[2] - self.x[2] * self.y[0];
        const a2 = self.x[0] * self.y[3] - self.x[3] * self.y[0];
        const a3 = self.x[1] * self.y[2] - self.x[2] * self.y[1];
        const a4 = self.x[1] * self.y[3] - self.x[3] * self.y[1];
        const a5 = self.x[2] * self.y[3] - self.x[3] * self.y[2];
        const b0 = self.z[0] * self.w[1] - self.z[1] * self.w[0];
        const b1 = self.z[0] * self.w[2] - self.z[2] * self.w[0];
        const b2 = self.z[0] * self.w[3] - self.z[3] * self.w[0];
        const b3 = self.z[1] * self.w[2] - self.z[2] * self.w[1];
        const b4 = self.z[1] * self.w[3] - self.z[3] * self.w[1];
        const b5 = self.z[2] * self.w[3] - self.z[3] * self.w[2];

        const det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;
        if (det == 0) {
            return null; // Not invertible
        }
        const inv_det = 1.0 / det;

        return .{
            .x = .{
                (self.y[1] * b5 - self.y[2] * b4 + self.y[3] * b3) * inv_det,
                (-self.x[1] * b5 + self.x[2] * b4 - self.x[3] * b3) * inv_det,
                (self.w[1] * a5 - self.w[2] * a4 + self.w[3] * a3) * inv_det,
                (-self.z[1] * a5 + self.z[2] * a4 - self.z[3] * a3) * inv_det,
            },
            .y = .{
                (-self.y[0] * b5 + self.y[2] * b2 - self.y[3] * b1) * inv_det,
                (self.x[0] * b5 - self.x[2] * b2 + self.x[3] * b1) * inv_det,
                (-self.w[0] * a5 + self.w[2] * a2 - self.w[3] * a1) * inv_det,
                (self.z[0] * a5 - self.z[2] * a2 + self.z[3] * a1) * inv_det,
            },
            .z = .{
                (self.y[0] * b4 - self.y[1] * b2 + self.y[3] * b0) * inv_det,
                (-self.x[0] * b4 + self.x[1] * b2 - self.x[3] * b0) * inv_det,
                (self.w[0] * a4 - self.w[1] * a2 + self.w[3] * a0) * inv_det,
                (-self.z[0] * a4 + self.z[1] * a2 - self.z[3] * a0) * inv_det,
            },
            .w = .{
                (-self.y[0] * b3 + self.y[1] * b1 - self.y[2] * b0) * inv_det,
                (self.x[0] * b3 - self.x[1] * b1 + self.x[2] * b0) * inv_det,
                (-self.w[0] * a3 + self.w[1] * a1 - self.w[2] * a0) * inv_det,
                (self.z[0] * a3 - self.z[1] * a1 + self.z[2] * a0) * inv_det,
            },
        };
    }

    pub fn transpose(self: *const Self) Mat4x4 {
        return .{
            .x = .{ self.x[0], self.y[0], self.z[0], self.w[0] },
            .y = .{ self.x[1], self.y[1], self.z[1], self.w[1] },
            .z = .{ self.x[2], self.y[2], self.z[2], self.w[2] },
            .w = .{ self.x[3], self.y[3], self.z[3], self.w[3] },
        };
    }
};
