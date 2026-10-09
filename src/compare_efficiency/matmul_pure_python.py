import random

Matrix = list[list[float]]


def matmul_pure_python(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    n = len(matrix_a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += matrix_a[i][k] * matrix_b[k][j]
            result[i][j] = total
    return result


def random_list_matrix(n: int, seed: int) -> Matrix:
    random_generator = random.Random(seed)
    return [[random_generator.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
