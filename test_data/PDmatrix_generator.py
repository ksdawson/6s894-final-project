#!/usr/bin/env python3

import numpy as np
import os

np.random.seed(0xca7cafe)
script_dir = os.path.dirname(os.path.realpath(__file__))

def generate_SPD_matrix(n):
    """
    Generate a dense n x n symmetric positive definite matrix.
    """
    A = np.random.rand(n, n).astype(np.float32)  # generate a random n x n matrix

    # Option 1 (faster, O(n^2)): make symmetric
    A = 0.5 * (A + A.T).astype(np.float32)

    # Option 2 (O(n^3)): form A * A^T
    # A = A @ A.T

    # Ensure positive definiteness by adding nI
    A += n * np.eye(n).astype(np.float32)

    return A

def save_matrix_to_bin(A, filename):
    """
    Save matrix A to a binary file in row-major order.
    A: numpy array (2D matrix)
    filename: path to output binary file
    """
    # Ensure the matrix is in C-contiguous (row-major) order
    # A = np.ascontiguousarray(A, dtype=np.float32)

    A = A.astype(np.float32)
    filepath = os.path.join(script_dir, filename)
    print(A)
    # Write to binary file
    with open(filepath, "wb") as f:
        f.write(A.tobytes())
    print(f"Wrote {filepath!r}")

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix A.
    A: numpy array (2D matrix)
    return: numpy array (2D matrix)
    """
    L = np.linalg.cholesky(A)
    return L

def read_matrix_from_bin(filename):
    """
    Read a matrix from a binary file.
    filename: path to input binary file
    return: numpy array (2D matrix)
    """
    A = np.fromfile(filename, dtype=np.float32)
    return A.reshape(n, n)

def backup(size_i, size_j, size_k):
    a = (np.random.randn(size_i, size_k) / size_k**0.5).astype(np.float32)
    b = np.random.randn(size_k, size_j).astype(np.float32)
    c = a @ b

    prefix = os.path.join(script_dir, f"test_{size_i}x{size_j}x{size_k}")
    a_fname = f"{prefix}_a.bin"
    b_fname = f"{prefix}_b.bin"
    c_fname = f"{prefix}_c.bin"

    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")

    with open(b_fname, "wb") as f:
        f.write(b.tobytes())
    print(f"Wrote {b_fname!r}")

    with open(c_fname, "wb") as f:
        f.write(c.tobytes())
    print(f"Wrote {c_fname!r}")

def write_example(n):
    # a = (np.random.randn(size_i, size_k) / size_k**0.5).astype(np.float32)
    # b = np.random.randn(size_k, size_j).astype(np.float32)
    # c = a @ b
    a = np.random.rand(n, n).astype(np.float32)  # generate a random n x n matrix
    # Option 1 (faster, O(n^2)): make symmetric
    a = 0.5 * (a + a.T).astype(np.float32)
    # Ensure positive definiteness by adding nI
    a += n * np.eye(n).astype(np.float32)

    prefix = os.path.join(script_dir, f"test_{n}x{n}x{n}")
    a_fname = f"{prefix}_a.bin"

    # prefix = os.path.join(script_dir, f"cholesky_{n}x{n}")
    # a_fname = f"{prefix}_a.bin"
    # b_fname = f"{prefix}_b.bin"
    # c_fname = f"{prefix}_c.bin"

    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")

    # with open(b_fname, "wb") as f:
    #     f.write(b.tobytes())
    # print(f"Wrote {b_fname!r}")

    # with open(c_fname, "wb") as f:
    #     f.write(c.tobytes())
    # print(f"Wrote {c_fname!r}")




if __name__ == "__main__":
    # backup(256, 256, 256)
    # backup(3072, 3072, 3072)
    # write_example(256)
    # n = [256]
    # for n in n:
    #     write_example(n)
    #     # A = generate_SPD_matrix(n)
    #     # save_matrix_to_bin(A, f"pdmatrix_{n}x{n}.bin")
    #     # L = cholesky_decomposition(A)
    #     # save_matrix_to_bin(L, f"cholesky_{n}x{n}.bin")
    #     # # A = read_matrix_from_bin(f"PDmatrix_{n}x{n}.bin")

    filename = "test_256x256x256_a.bin"

    # Try reading as float32 (4 bytes per element)
    data = np.fromfile(filename, dtype=np.float32)

    print(f"File: {filename}")
    print(f"Total elements read: {len(data)}")
    print(f"File size: {len(data) * 4} bytes")

    # Try to infer dimensions
    # If it's 256x256, should have 65536 elements
    if len(data) == 256 * 256:
        print("Detected as 256x256 matrix")
        matrix = data.reshape(256, 256)
        print(f"Matrix shape: {matrix.shape}")
        
        # Print first 5x5 block
        print("\nFirst 5x5 block:")
        print(matrix[:5, :5])
        
        # Print statistics
        print(f"\nMin: {matrix.min():.6f}")
        print(f"Max: {matrix.max():.6f}")
        print(f"Mean: {matrix.mean():.6f}")
        print(f"Std: {matrix.std():.6f}")