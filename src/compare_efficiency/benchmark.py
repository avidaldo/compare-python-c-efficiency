import statistics
import time
from collections.abc import Callable

import numpy as np


def random_matrices(n: int, dtype: type = np.float64, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(seed)
    matrix_a = random_generator.standard_normal((n, n)).astype(dtype)
    matrix_b = random_generator.standard_normal((n, n)).astype(dtype)
    return matrix_a, matrix_b


def time_function(function: Callable[[], object], repeats: int = 5, warmup: int = 1,
                  synchronize: Callable[[], None] | None = None) -> float:
    """Median wall-clock seconds of `function()`. `synchronize` waits for asynchronous (GPU) work to finish."""
    for _ in range(warmup):
        function()
    if synchronize is not None:
        synchronize()

    measured_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if synchronize is not None:
            synchronize()
        measured_seconds.append(time.perf_counter() - start)
    return statistics.median(measured_seconds)


def matmul_flops(n: int) -> int:
    # n^2 output elements, each needing n multiplications and n additions
    return 2 * n**3


def gflops(n: int, seconds: float) -> float:
    return matmul_flops(n) / seconds / 1e9
