const std = @import("std");
const vectorium = @import("vectorium");

const vector = vectorium.core.vector;
const matrix = vectorium.core.matrix;
const functions = vectorium.core.functions;
const print = std.debug.print;

pub fn main() !void {
    // Trying out some cool vector theorems and problems
    print("\n=== Vector Theorems and Problems ===\n\n", .{});

    // 1. Demonstrate triangle inequality: |a + b| <= |a| + |b|
    print("1. Triangle Inequality Theorem:\n", .{});
    const a = vector.Vec3{ .x = 3.0, .y = 4.0, .z = 0.0 };
    const b = vector.Vec3{ .x = 1.0, .y = 2.0, .z = 2.0 };
    const a_b = a.add(&b);
    const left_side = a_b.length();
    const right_side = a.length() + b.length();
    print("   |a + b| = {d:.3}, |a| + |b| = {d:.3}\n", .{ left_side, right_side });
    print("   Triangle inequality holds: {}\n\n", .{left_side <= right_side});

    // 2. Demonstrate Cauchy-Schwarz inequality: |a · b| <= |a| * |b|
    print("2. Cauchy-Schwarz Inequality:\n", .{});
    const dot_ab = @abs(a.dot(&b));
    const product_lengths = a.length() * b.length();
    print("   |a · b| = {d:.3}, |a| * |b| = {d:.3}\n", .{ dot_ab, product_lengths });
    print("   Cauchy-Schwarz holds: {}\n\n", .{dot_ab <= product_lengths});

    // 3. Cross product creates orthogonal vector
    print("3. Cross Product Orthogonality:\n", .{});
    const cross_result = a.cross(&b);
    const dot_a_cross = a.dot(&cross_result);
    const dot_b_cross = b.dot(&cross_result);
    print("   a × b = ({d:.3}, {d:.3}, {d:.3})\n", .{ cross_result.x, cross_result.y, cross_result.z });
    print("   a · (a × b) = {d:.6} (should be ~0)\n", .{dot_a_cross});
    print("   b · (a × b) = {d:.6} (should be ~0)\n\n", .{dot_b_cross});

    // 4. BAC-CAB rule: a × (b × c) = b(a·c) - c(a·b)
    print("4. BAC-CAB Vector Triple Product:\n", .{});
    const c = vector.Vec3{ .x = 0.0, .y = 0.0, .z = 1.0 };
    const b_cross_c = b.cross(&c);
    const left = a.cross(&b_cross_c);
    const a_dot_c = a.dot(&c);
    const a_dot_b = a.dot(&b);
    const right = b.scale(a_dot_c).sub(&c.scale(a_dot_b));
    print("   a × (b × c) = ({d:.3}, {d:.3}, {d:.3})\n", .{ left.x, left.y, left.z });
    print("   b(a·c) - c(a·b) = ({d:.3}, {d:.3}, {d:.3})\n\n", .{ right.x, right.y, right.z });

    // 5. Projection of vector a onto b
    print("5. Vector Projection:\n", .{});
    const unit_i = vector.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const vec_to_project = vector.Vec3{ .x = 3.0, .y = 4.0, .z = 0.0 };
    const projection = functions.projectVec3(&vec_to_project, &unit_i);
    print("   Projecting ({d}, {d}, {d}) onto x-axis\n", .{ vec_to_project.x, vec_to_project.y, vec_to_project.z });
    print("   Projection: ({d:.3}, {d:.3}, {d:.3})\n\n", .{ projection.x, projection.y, projection.z });

    print("\n=== Linear Transformations Demo ===\n\n", .{});

    // 6. 2D Rotation
    print("6. 2D Rotation (90 degrees):\n", .{});
    const rot_90 = functions.rotation2D(std.math.pi / 2.0);
    const v2d = vector.Vec2{ .x = 1.0, .y = 0.0 };
    const rotated_2d = functions.transform2D(&rot_90, &v2d);
    print("   Original: ({d:.3}, {d:.3})\n", .{ v2d.x, v2d.y });
    print("   Rotated:  ({d:.3}, {d:.3})\n\n", .{ rotated_2d.x, rotated_2d.y });

    // 7. 3D Rotation around axis
    print("7. 3D Rotation around Y-axis (45 degrees):\n", .{});
    const rot_y = functions.rotationY(std.math.pi / 4.0);
    const v3d = vector.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const rotated_3d = functions.transform3D(&rot_y, &v3d);
    print("   Original: ({d:.3}, {d:.3}, {d:.3})\n", .{ v3d.x, v3d.y, v3d.z });
    print("   Rotated:  ({d:.3}, {d:.3}, {d:.3})\n\n", .{ rotated_3d.x, rotated_3d.y, rotated_3d.z });

    // 8. Gram-Schmidt Orthogonalization
    print("8. Gram-Schmidt Orthogonalization:\n", .{});
    const gs_v1 = vector.Vec3{ .x = 1.0, .y = 1.0, .z = 0.0 };
    const gs_v2 = vector.Vec3{ .x = 1.0, .y = 0.0, .z = 1.0 };
    const gs_v3 = vector.Vec3{ .x = 0.0, .y = 1.0, .z = 1.0 };
    const orthonormal = functions.gramSchmidt3D(gs_v1, gs_v2, gs_v3);
    print("   Input vectors:\n", .{});
    print("   v1 = ({d:.3}, {d:.3}, {d:.3})\n", .{ gs_v1.x, gs_v1.y, gs_v1.z });
    print("   v2 = ({d:.3}, {d:.3}, {d:.3})\n", .{ gs_v2.x, gs_v2.y, gs_v2.z });
    print("   v3 = ({d:.3}, {d:.3}, {d:.3})\n", .{ gs_v3.x, gs_v3.y, gs_v3.z });
    print("   Orthonormal basis:\n", .{});
    print("   e1 = ({d:.3}, {d:.3}, {d:.3}), length = {d:.3}\n", .{ orthonormal[0].x, orthonormal[0].y, orthonormal[0].z, orthonormal[0].length() });
    print("   e2 = ({d:.3}, {d:.3}, {d:.3}), length = {d:.3}\n", .{ orthonormal[1].x, orthonormal[1].y, orthonormal[1].z, orthonormal[1].length() });
    print("   e3 = ({d:.3}, {d:.3}, {d:.3}), length = {d:.3}\n", .{ orthonormal[2].x, orthonormal[2].y, orthonormal[2].z, orthonormal[2].length() });
    print("   e1 · e2 = {d:.6} (should be ~0)\n", .{orthonormal[0].dot(&orthonormal[1])});
    print("   e1 · e3 = {d:.6} (should be ~0)\n", .{orthonormal[0].dot(&orthonormal[2])});
    print("   e2 · e3 = {d:.6} (should be ~0)\n\n", .{orthonormal[1].dot(&orthonormal[2])});

    // 9. Matrix determinant and inverse
    print("9. Matrix Determinant and Inverse:\n", .{});
    const m3 = matrix.Mat3x3{
        .x = .{ 1.0, 2.0, 3.0 },
        .y = .{ 0.0, 1.0, 4.0 },
        .z = .{ 5.0, 6.0, 0.0 },
    };
    const det = m3.determinant();
    print("   Matrix determinant: {d:.3}\n", .{det});
    if (m3.inverse()) |inv| {
        const identity_check = m3.mul(&inv);
        print("   Inverse exists!\n", .{});
        print("   M * M^-1 diagonal: ({d:.3}, {d:.3}, {d:.3})\n", .{ identity_check.x[0], identity_check.y[1], identity_check.z[2] });
    }
    print("\n", .{});

    // 10. Angle between vectors
    print("10. Angle Between Vectors:\n", .{});
    const angle_v1 = vector.Vec3{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const angle_v2 = vector.Vec3{ .x = 1.0, .y = 1.0, .z = 0.0 };
    const angle = functions.angleBetween3D(&angle_v1, &angle_v2);
    print("   v1 = ({d:.3}, {d:.3}, {d:.3})\n", .{ angle_v1.x, angle_v1.y, angle_v1.z });
    print("   v2 = ({d:.3}, {d:.3}, {d:.3})\n", .{ angle_v2.x, angle_v2.y, angle_v2.z });
    print("   Angle: {d:.3} radians ({d:.1} degrees)\n\n", .{ angle, angle * 180.0 / std.math.pi });

    // 11. Linear interpolation
    print("11. Linear Interpolation (LERP):\n", .{});
    const lerp_start = vector.Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const lerp_end = vector.Vec3{ .x = 10.0, .y = 20.0, .z = 30.0 };
    const lerp_25 = functions.lerp3D(&lerp_start, &lerp_end, 0.25);
    const lerp_50 = functions.lerp3D(&lerp_start, &lerp_end, 0.5);
    const lerp_75 = functions.lerp3D(&lerp_start, &lerp_end, 0.75);
    print("   Start: ({d:.1}, {d:.1}, {d:.1})\n", .{ lerp_start.x, lerp_start.y, lerp_start.z });
    print("   25%:   ({d:.1}, {d:.1}, {d:.1})\n", .{ lerp_25.x, lerp_25.y, lerp_25.z });
    print("   50%:   ({d:.1}, {d:.1}, {d:.1})\n", .{ lerp_50.x, lerp_50.y, lerp_50.z });
    print("   75%:   ({d:.1}, {d:.1}, {d:.1})\n", .{ lerp_75.x, lerp_75.y, lerp_75.z });
    print("   End:   ({d:.1}, {d:.1}, {d:.1})\n\n", .{ lerp_end.x, lerp_end.y, lerp_end.z });

    print("=== All tests complete! ===\n", .{});
}

pub fn test2DLogic() void {
    // Vector along X axis (unit vector i)
    const vi_2d = vector.Vec2{
        .x = 1.0,
        .y = 0.0,
    };

    // Vector along Y axis (unit vector j)
    const varj_2d = vector.Vec2{
        .x = 0.0,
        .y = 1.0,
    };

    // Test addition
    const sum = vi_2d.add(varj_2d);
    print("2D Addition: ({d}, {d})\n", .{ sum.x, sum.y });

    // Test subtraction
    const diff = vi_2d.sub(varj_2d);
    print("2D Subtraction: ({d}, {d})\n", .{ diff.x, diff.y });

    // Test scalar multiplication
    const scaled = vi_2d.scale(2.0);
    print("2D Scaled: ({d}, {d})\n", .{ scaled.x, scaled.y });

    // Test dot product
    const dot_result = vi_2d.dot(varj_2d);
    print("2D Dot Product: {d}\n", .{dot_result});

    // Test length
    const length = vi_2d.length();
    print("2D Length: {d}\n", .{length});
}

pub fn test3DLogic() void {
    // Vector along X axis (unit vector i)
    const veci_3d = vector.Vec3{
        .x = 1.0,
        .y = 0.0,
        .z = 0.0,
    };

    // Vector along Y axis (unit vector j)
    const vecj_3d = vector.Vec3{
        .x = 0.0,
        .y = 1.0,
        .z = 0.0,
    };

    // // Vector along Z axis (unit vector k)
    // const veck_3d = vector.Vec3{
    //     .x = 0.0,
    //     .y = 0.0,
    //     .z = 1.0,
    // };

    // Test addition
    const sum = veci_3d.add(vecj_3d);
    print("3D Addition: ({d}, {d}, {d})\n", .{ sum.x, sum.y, sum.z });

    // Test subtraction
    const diff = veci_3d.sub(vecj_3d);
    print("3D Subtraction: ({d}, {d}, {d})\n", .{ diff.x, diff.y, diff.z });

    // Test scalar multiplication
    const scaled = veci_3d.scale(2.0);
    print("3D Scaled: ({d}, {d}, {d})\n", .{ scaled.x, scaled.y, scaled.z });

    // Test dot product
    const dot_result = veci_3d.dot(vecj_3d);
    print("3D Dot Product: {d}\n", .{dot_result});

    // Test length
    const length = veci_3d.length();
    print("3D Length: {d}\n", .{length});

    // Test cross product (unique to 3D)
    const cross_result = veci_3d.cross(vecj_3d);
    print("3D Cross Product (i x j): ({d}, {d}, {d})\n", .{ cross_result.x, cross_result.y, cross_result.z });
}

pub fn test4DLogic() void {
    // Vector along X axis (unit vector i)
    const veci_4d = vector.Vec4{
        .x = 1.0,
        .y = 0.0,
        .z = 0.0,
        .w = 0.0,
    };

    // Vector along Y axis (unit vector j)
    const vecj_4d = vector.Vec4{
        .x = 0.0,
        .y = 1.0,
        .z = 0.0,
        .w = 0.0,
    };

    // // Vector along Z axis (unit vector k)
    // const veck_4d = vector.Vec4{
    //     .x = 0.0,
    //     .y = 0.0,
    //     .z = 1.0,
    //     .w = 0.0,
    // };
    //
    // // Vector along W axis (unit vector l)
    // const vecl_4d = vector.Vec4{
    //     .x = 0.0,
    //     .y = 0.0,
    //     .z = 0.0,
    //     .w = 1.0,
    // };

    // Test addition
    const sum = veci_4d.add(vecj_4d);
    print("4D Addition: ({d}, {d}, {d}, {d})\n", .{ sum.x, sum.y, sum.z, sum.w });

    // Test subtraction
    const diff = veci_4d.sub(vecj_4d);
    print("4D Subtraction: ({d}, {d}, {d}, {d})\n", .{ diff.x, diff.y, diff.z, diff.w });

    // Test scalar multiplication
    const scaled = veci_4d.scale(2.0);
    print("4D Scaled: ({d}, {d}, {d}, {d})\n", .{ scaled.x, scaled.y, scaled.z, scaled.w });

    // Test dot product
    const dot_result = veci_4d.dot(vecj_4d);
    print("4D Dot Product: {d}\n", .{dot_result});

    // Test length
    const length = veci_4d.length();
    print("4D Length: {d}\n", .{length});
}

// test "simple test" {
//     const gpa = std.testing.allocator;
//     var list: std.ArrayList(i32) = .empty;
//     defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
//     try list.append(gpa, 42);
//     try std.testing.expectEqual(@as(i32, 42), list.pop());
// }
//
// test "fuzz example" {
//     const Context = struct {
//         fn testOne(context: @This(), input: []const u8) anyerror!void {
//             _ = context;
//             // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
//             try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
//         }
//     };
//     try std.testing.fuzz(Context{}, Context.testOne, .{});
// }
