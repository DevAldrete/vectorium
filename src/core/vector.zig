//! Basic vector mathematics library in Zig

const std = @import("std");

// 2D Vector struct with basic operations
pub const Vec2 = struct {
    const Self = @This();
    x: f32,
    y: f32,

    pub fn add(self: *const Self, b: *const Vec2) Vec2 {
        return .{
            .x = self.x + b.x,
            .y = self.y + b.y,
        };
    }

    pub fn sub(self: *const Self, b: *const Vec2) Vec2 {
        return .{
            .x = self.x - b.x,
            .y = self.y - b.y,
        };
    }

    pub fn dot(self: *const Self, b: *const Vec2) f32 {
        return self.x * b.x + self.y * b.y;
    }

    pub fn length(self: *const Self) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }

    pub fn normalize(self: *const Self) Self {
        const len = self.length();
        if (len != 0) {
            return .{
                .x = self.x / len,
                .y = self.y / len,
            };
        }
        return .{ .x = 0, .y = 0 };
    }

    pub fn scale(self: *const Self, scalar: f32) Self {
        return .{
            .x = self.x * scalar,
            .y = self.y * scalar,
        };
    }
};

// 3D Vector struct with basic operations
pub const Vec3 = struct {
    const Self = @This();
    x: f32,
    y: f32,
    z: f32,

    pub fn add(self: *const Self, b: *const Vec3) Vec3 {
        return .{
            .x = self.x + b.x,
            .y = self.y + b.y,
            .z = self.z + b.z,
        };
    }

    pub fn sub(self: *const Self, b: *const Vec3) Vec3 {
        return .{
            .x = self.x - b.x,
            .y = self.y - b.y,
            .z = self.z - b.z,
        };
    }

    pub fn dot(self: *const Self, b: *const Vec3) f32 {
        return self.x * b.x + self.y * b.y + self.z * b.z;
    }

    pub fn scale(self: *const Self, scalar: f32) Self {
        return .{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
        };
    }

    pub fn length(self: *const Self) f32 {
        return @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    pub fn normalize(self: *const Self) Vec3 {
        const len = self.length();
        if (len != 0) {
            return .{
                .x = self.x / len,
                .y = self.y / len,
                .z = self.z / len,
            };
        }
        return .{ .x = 0, .y = 0, .z = 0 };
    }

    pub fn cross(self: *const Self, b: *const Vec3) Vec3 {
        return .{
            .x = self.y * b.z - self.z * b.y,
            .y = self.z * b.x - self.x * b.z,
            .z = self.x * b.y - self.y * b.x,
        };
    }
};

// 4D Vector struct with basic operations
pub const Vec4 = struct {
    const Self = @This();
    x: f32,
    y: f32,
    z: f32,
    w: f32,

    pub fn add(self: *const Self, b: *const Vec4) Vec4 {
        return .{
            .x = self.x + b.x,
            .y = self.y + b.y,
            .z = self.z + b.z,
            .w = self.w + b.w,
        };
    }

    pub fn sub(self: *const Self, b: *const Vec4) Vec4 {
        return .{
            .x = self.x - b.x,
            .y = self.y - b.y,
            .z = self.z - b.z,
            .w = self.w - b.w,
        };
    }

    pub fn dot(self: *const Self, b: *const Vec4) f32 {
        return self.x * b.x + self.y * b.y + self.z * b.z + self.w * b.w;
    }

    pub fn length(self: *const Self) f32 {
        return @sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w);
    }

    pub fn normalize(self: *const Self) Self {
        const len = self.length();
        if (len != 0) {
            return .{
                .x = self.x / len,
                .y = self.y / len,
                .z = self.z / len,
                .w = self.w / len,
            };
        }
        return .{ .x = 0, .y = 0, .z = 0, .w = 0 };
    }

    pub fn scale(self: *const Self, scalar: f32) Self {
        return .{
            .x = self.x * scalar,
            .y = self.y * scalar,
            .z = self.z * scalar,
            .w = self.w * scalar,
        };
    }
};

// SIMD version (placeholder)
// pub const Vec2SIMD = struct {
//     const Self = @This();
//     a: []f32,
//     b: []f32,
// };
