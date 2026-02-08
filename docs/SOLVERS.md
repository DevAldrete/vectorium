# Vectorium Solvers

Comprehensive linear algebra solver system for Vectorium.

## Overview

The solver system provides a well-architected, scalable foundation for solving various linear algebra problems. The design follows key principles:

1. **Separation of Concerns**: Different solver types are organized into focused submodules
2. **Strategy Pattern**: Multiple algorithms available for the same problem type
3. **Extensibility**: Easy to add new solvers and extend existing ones
4. **Type Safety**: Leverages Zig's compile-time capabilities
5. **Performance**: Designed with numerical stability and efficiency in mind

## Architecture

```
src/core/solvers.zig          # Main module with common utilities
├── solvers/linear_systems.zig  # Linear equation solvers
├── solvers/decomposition.zig   # Matrix factorization methods
└── solvers/eigen.zig           # Eigenvalue/eigenvector solvers
```

## Module: Linear Systems (`solvers.linear_systems`)

Solves systems of linear equations: **Ax = b**

### Available Solvers

#### Gaussian Elimination with Partial Pivoting
- **Functions**: `solve2x2()`, `solve3x3()`, `solve4x4()`
- **Complexity**: O(n³)
- **Best for**: General dense systems, one-time solutions
- **Features**: Numerical stability through pivoting

```zig
const A = matrix.Mat3x3{
    .x = .{ 2.0, 1.0, 1.0 },
    .y = .{ 1.0, 3.0, 2.0 },
    .z = .{ 1.0, 1.0, 4.0 },
};
const b = vector.Vec3{ .x = 5.0, .y = 8.0, .z = 10.0 };
const x = try solvers.linear_systems.solve3x3(A, b);
```

### Future Extensions
- Iterative methods (Jacobi, Gauss-Seidel, Conjugate Gradient)
- Sparse matrix solvers
- Overdetermined/underdetermined systems (least squares)
- Banded matrix solvers

## Module: Decomposition (`solvers.decomposition`)

Matrix factorization techniques used as building blocks for other algorithms.

### LU Decomposition
Factors **A = PLU** where:
- **P** is a permutation matrix (for numerical stability)
- **L** is lower triangular
- **U** is upper triangular

**Function**: `lu3x3()`

**Use cases**:
- Solving multiple systems with the same coefficient matrix
- Computing determinants efficiently
- Matrix inversion

```zig
const lu = try solvers.decomposition.lu3x3(A);
const x = try lu.solve(b);           // Solve Ax = b
const det = lu.determinant();         // Compute det(A)
```

### QR Decomposition
Factors **A = QR** where:
- **Q** is orthogonal (Q^T * Q = I)
- **R** is upper triangular

**Function**: `qr3x3()`

**Method**: Gram-Schmidt orthogonalization

**Use cases**:
- Least squares problems
- Computing eigenvalues (QR algorithm)
- Orthonormalization

```zig
const qr = try solvers.decomposition.qr3x3(A);
const x = try qr.solveLeastSquares(b);
```

### Future Extensions
- **SVD** (Singular Value Decomposition) - pseudoinverses, data compression
- **Cholesky** decomposition - symmetric positive definite matrices
- **Schur** decomposition - computing matrix functions
- **Householder** reflections - more stable QR

## Module: Eigenvalues (`solvers.eigen`)

Computes eigenvalues and eigenvectors: **Av = λv**

### Power Iteration
Finds the **dominant eigenvalue** (largest absolute value).

**Function**: `powerIteration3x3()`

**Algorithm**: Repeatedly multiply by matrix and normalize

**Complexity**: O(n² * iterations)

**Best for**: Finding largest eigenvalue quickly

```zig
const config = solvers.IterativeConfig{
    .max_iterations = 1000,
    .tolerance = 1e-6,
    .verbose = false,
};
const result = try solvers.eigen.powerIteration3x3(A, config);
// result.eigenvalue, result.eigenvector, result.converged
```

### Inverse Power Iteration
Finds the **smallest eigenvalue**.

**Function**: `inversePowerIteration3x3()`

**Algorithm**: Apply power iteration to A^(-1)

**Best for**: Finding smallest eigenvalue

```zig
const result = try solvers.eigen.inversePowerIteration3x3(A, config);
```

### QR Algorithm
Finds **all eigenvalues** simultaneously.

**Function**: `qrAlgorithm3x3()`

**Algorithm**: Iteratively apply QR decomposition

**Complexity**: O(n³ * iterations)

**Best for**: Computing full eigenspectrum

```zig
const result = try solvers.eigen.qrAlgorithm3x3(A, config);
// result.eigenvalues (array of 3), result.converged
```

### Rayleigh Quotient
Estimates eigenvalue from an approximate eigenvector.

**Function**: `rayleighQuotient3x3()`

**Formula**: λ ≈ (v^T * A * v) / (v^T * v)

### Future Extensions
- **Jacobi method** - for symmetric matrices
- **Arnoldi iteration** - for large sparse matrices
- **Lanczos algorithm** - symmetric sparse matrices
- **Eigenvector computation** from QR algorithm
- **Shifted inverse iteration** - find eigenvalue near a target

## Common Utilities

### Error Types
```zig
pub const SolverError = error{
    SingularMatrix,         // Matrix is not invertible
    NoConvergence,          // Iterative method didn't converge
    InvalidInput,           // Invalid parameters
    NumericalInstability,   // Computation became unstable
    DimensionMismatch,      // Matrix/vector size mismatch
};
```

### Iterative Configuration
```zig
pub const IterativeConfig = struct {
    max_iterations: usize = 1000,
    tolerance: f32 = 1e-6,
    verbose: bool = false,
};
```

### Numerical Utilities
- `isNearZero(value, epsilon)` - Check if value is approximately zero
- `isClose(a, b, rel_tol, abs_tol)` - Check if two values are close

## Design Patterns for Scalability

### 1. Strategy Pattern
Multiple algorithms for the same problem type allows choosing the best approach:
- Direct vs iterative methods
- Different decompositions for different use cases

### 2. Composition
Solvers build on each other:
- QR algorithm uses QR decomposition
- Inverse power iteration uses linear system solver
- LU decomposition enables efficient multiple solves

### 3. Generic Foundation
The architecture easily extends to:
- Arbitrary dimensions (with generic programming)
- Different numeric types (f64, complex numbers)
- Sparse matrices (with abstract interfaces)

### 4. Future-Ready Structure
Each submodule has clearly documented extension points:
- More sophisticated algorithms
- Specialized variants
- Performance optimizations (SIMD, GPU)

## Testing

All solvers include comprehensive unit tests:

```bash
# Run all tests
zig build test

# Run tests for a specific module
zig test src/core/solvers/linear_systems.zig
zig test src/core/solvers/decomposition.zig
zig test src/core/solvers/eigen.zig
```

## Example Usage

See `examples/solver_demo.zig` for a comprehensive demonstration:

```bash
zig build solver-demo
```

## Performance Considerations

### Current Implementation
- **Target**: Correctness and clarity
- **Optimizations**: Partial pivoting, numerical stability checks
- **Size**: Small to medium matrices (2x2 to 4x4)

### Future Optimizations
1. **SIMD**: Vectorize inner loops
2. **Blocked algorithms**: Better cache utilization for larger matrices
3. **Sparse**: Special handling for sparse matrices
4. **Parallel**: Multi-threaded decompositions
5. **GPU**: Offload to GPU for very large systems

## Numerical Stability

The implementation includes several stability features:

1. **Partial pivoting** in Gaussian elimination
2. **Tolerance checks** for near-zero pivots
3. **Gram-Schmidt** (with option to upgrade to modified GS)
4. **Normalization** in iterative methods

## Extension Guide

### Adding a New Solver Type

1. Create new file: `src/core/solvers/new_solver.zig`
2. Implement solver functions with proper error handling
3. Add comprehensive tests
4. Export from `src/core/solvers.zig`
5. Document in this README

### Adding Support for Larger Matrices

The current structure supports extension to arbitrary sizes:

```zig
// Example: Generic solver
pub fn solveNxN(comptime n: usize, A: MatNxN(n), b: VecN(n)) !VecN(n) {
    // Implementation using generic matrix types
}
```

### Adding New Algorithms

Each submodule has clearly marked extension points. For example:

```zig
// In linear_systems.zig
pub fn solveJacobi3x3(A: matrix.Mat3x3, b: vector.Vec3, 
                      config: IterativeConfig) !IterativeResult(vector.Vec3) {
    // Implement Jacobi iteration
}
```

## References

The implementations follow standard numerical linear algebra algorithms from:
- Golub & Van Loan, "Matrix Computations"
- Trefethen & Bau, "Numerical Linear Algebra"
- Demmel, "Applied Numerical Linear Algebra"
