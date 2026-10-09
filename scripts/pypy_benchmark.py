"""Time the pure-Python matmul with whichever interpreter runs this script (CPython or PyPy). Prints JSON."""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_efficiency.matmul_pure_python import matmul_pure_python, random_list_matrix

SIZES = [64, 128, 256]
REPEATS = 3


def median_seconds(n: int) -> float:
    matrix_a = random_list_matrix(n, seed=1)
    matrix_b = random_list_matrix(n, seed=2)
    # Warm-up run: gives PyPy's JIT the chance to compile the hot loop before we measure
    matmul_pure_python(matrix_a, matrix_b)
    measured_seconds = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        matmul_pure_python(matrix_a, matrix_b)
        measured_seconds.append(time.perf_counter() - start)
    return sorted(measured_seconds)[REPEATS // 2]


results = {
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "seconds": {n: median_seconds(n) for n in SIZES},
}
print(json.dumps(results))
