import numpy as np

def generate_SPD_matrix(n):
    """
    Generate a dense n x n symmetric positive definite matrix.
    """
    A = np.random.rand(n, n)  # generate a random n x n matrix

    # Option 1 (faster, O(n^2)): make symmetric
    A = 0.5 * (A + A.T)

    # Option 2 (O(n^3)): form A * A^T
    # A = A @ A.T

    # Ensure positive definiteness by adding nI
    A += n * np.eye(n)

    return A

def save_matrix_to_bin(A, filename):
    """
    Save matrix A to a binary file in row-major order.
    A: numpy array (2D matrix)
    filename: path to output binary file
    """
    # Ensure the matrix is in C-contiguous (row-major) order
    A = np.ascontiguousarray(A, dtype=np.float32)
    # Write to binary file
    A.tofile(filename)

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix A.
    A: numpy array (2D matrix)
    return: numpy array (2D matrix)
    """
    L = np.linalg.cholesky(A)
    return L


if __name__ == "__main__":
    n = [64, 128, 256, 512, 1024, 2048, 4096]
    for n in n:
        A = generate_SPD_matrix(n)
        save_matrix_to_bin(A, f"PDmatrix_{n}x{n}.bin")
        L = cholesky_decomposition(A)
        save_matrix_to_bin(L, f"Cholesky_{n}x{n}.bin")