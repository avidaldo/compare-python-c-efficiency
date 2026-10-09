import ctypes
import subprocess
from pathlib import Path

import numpy as np

from compare_efficiency.project_paths import BUILD_DIRECTORY, C_DIRECTORY

C_SOURCE = C_DIRECTORY / "matmul.c"

DOUBLE_MATRIX = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


def compile_c_library(optimization_flags: list[str], name: str) -> Path:
    BUILD_DIRECTORY.mkdir(exist_ok=True)
    library_path = BUILD_DIRECTORY / f"libmatmul_{name}.so"
    command = ["gcc", *optimization_flags, "-shared", "-fPIC", str(C_SOURCE), "-o", str(library_path)]
    subprocess.run(command, check=True)
    return library_path


def load_c_library(library_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(library_path))
    for function_name in ("matmul_ijk", "matmul_ikj"):
        function = getattr(library, function_name)
        function.argtypes = [DOUBLE_MATRIX, DOUBLE_MATRIX, DOUBLE_MATRIX, ctypes.c_size_t]
        function.restype = None
    return library


def call_c_matmul(c_function: ctypes._CFuncPtr, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    n = matrix_a.shape[0]
    result = np.empty((n, n), dtype=np.float64)
    # ctypes hands C the raw address of each NumPy buffer: no copy, no Python objects per element
    c_function(matrix_a, matrix_b, result, n)
    return result
