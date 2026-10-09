# Why does AI use Python if it is slower than C?

Pure Python is one of the slowest mainstream languages, yet almost all AI work is done in Python. This project resolves that contradiction step by step, following one operation, **matrix multiplication**, from a naive Python loop to a graphics card.

It is written for students who know basic Python (variables, loops, functions, lists). No previous knowledge of NumPy, machine learning or computer architecture is needed; every concept is introduced when it is first used and collected in the [glossary](glossary.md).

## Path

Read these in order. Together they tell the whole story.

| File | What you learn |
|---|---|
| [`01_languages_for_ai`](01_languages_for_ai.ipynb) | Compiled and interpreted languages, which languages are used in AI, and why Python is the main one |
| [`02_matrices_in_ai`](02_matrices_in_ai.ipynb) | Vectors, matrices and tensors; matrix multiplication; why AI (including Transformers) is mostly matrix multiplication |
| [`03_python_vs_c`](03_python_vs_c.ipynb) | The same matrix multiplication in Python and in C: C is hundreds of times faster, and why |
| [`04_python_calling_c`](04_python_calling_c.ipynb) | NumPy is compiled C code called from Python; BLAS; Python as the glue language of AI |
| [`05_gpus_and_cuda`](05_gpus_and_cuda.ipynb) | Why GPUs made modern AI possible; CUDA as C for GPUs; cuBLAS |
| [`06_conclusion`](06_conclusion.md) | How and why Python drives AI through libraries built on other technologies |

## Deep dives

Optional notebooks for curious readers, in the [`deep-dives`](deep-dives) folder. Each one extends a notebook of the path, which never depends on them.

| Deep dive | Extends | What it adds |
|---|---|---|
| [How programs run](deep-dives/how_programs_run.ipynb) | 01 | Bytecode, static and dynamic typing, JIT and AOT compilation, machine code |
| [Numbers and memory](deep-dives/numbers_and_memory.ipynb) | 02, 03 | Floating-point numbers, how Python stores values, caches and cache lines |
| [Matrix multiplication in AI](deep-dives/matmul_in_ai.ipynb) | 02 | Image filters, attention in code, batched matrix multiplication, a real Transformer block measured with a profiler |
| [Loop order and the cache](deep-dives/loop_order_and_cache.ipynb) | 03 | Why reading memory in order matters so much; compiler flags; cache conflicts |
| [Inside BLAS](deep-dives/inside_blas.ipynb) | 04 | Cores and threads, the cost of each call, how an expert library organises its work |
| [Compiling Python](deep-dives/compiling_python.ipynb) | 04 | Numba, Cython and PyPy |
| [GPU performance](deep-dives/gpu_performance.ipynb) | 05 | Timing GPU code, data transfers, precision formats, the CUDA thread model, Triton |
| [Languages for AI beyond Python](deep-dives/language_landscape.md) | 01, 06 | C++, CUDA, Triton, JAX, Rust, Julia and Mojo; reference results |

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync                # everything except the GPU notebooks
uv sync --group gpu    # adds CuPy, needed for 05_gpus_and_cuda and deep-dives/gpu_performance
```

Without uv, create an environment with venv or conda and run `pip install -r requirements.txt`; it installs the same packages and the project's helpers (`-e .`). For the GPU notebooks, also install the CuPy line commented at the end of the file.

Requirements outside Python:

- **A C compiler (`gcc`)** for the C and Cython parts. Linux and macOS usually have one; on Windows, use WSL or Google Colab.
- **An NVIDIA GPU** for `05_gpus_and_cuda` and the GPU deep dive. Without one, upload the notebook to Google Colab and choose a GPU runtime; neither depends on other files of this project.
- **PyPy** is downloaded automatically by `uv` the first time the Compiling Python deep dive needs it.

PyTorch is a large download on Linux, because it bundles the GPU libraries.

## Project structure

```text
compare-python-c-efficiency/
├── 01-06                           # path
├── deep-dives/                     # optional deep dives
├── c/                              # C programs: sum_loop.c, type_error.c, matmul.c
├── src/compare_efficiency/         # helpers installed as a package by uv sync
│   ├── benchmark.py                # timing (warm-up + median), GFLOPS, random matrices
│   ├── c_library.py                # compiles c/matmul.c with gcc and loads it with ctypes
│   ├── matmul_pure_python.py       # naive pure-Python matrix multiplication
│   └── project_paths.py            # absolute paths, so notebooks work from any folder
├── scripts/pypy_benchmark.py       # the pure-Python benchmark, runnable under CPython or PyPy
└── glossary.md
```

## How measurements are made

Every implementation is checked against NumPy (`np.allclose`) before it is timed. Times are the median of several runs after a warm-up run, and GPU timings wait for the GPU to finish before stopping the clock. Speed is reported in GFLOPS (TFLOPS on the GPU), counting $2n^3$ operations per $n \times n$ product. Results vary between machines: compare ratios between implementations, not absolute numbers.
