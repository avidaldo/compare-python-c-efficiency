# Languages for AI beyond Python

## Layers, not rivals

The series leads to one conclusion: in AI, **Python is the language in which computations are described, and other languages are the ones in which they are executed.** So the useful question is not "which language is fastest?" but "which language is right for which **layer**?"

An AI system is built in layers, like a building. Each layer uses the one below it without needing to know its details:

```text
┌──────────────────────────────────────────────────────────────┐
│ Python: notebooks, training scripts, data pipelines, APIs    │  ← where most AI developers work
├──────────────────────────────────────────────────────────────┤
│ Frameworks: PyTorch, JAX, TensorFlow (Python API, C++ core)  │
├──────────────────────────────────────────────────────────────┤
│ Compilers and kernel languages: torch.compile, XLA, Triton   │
├──────────────────────────────────────────────────────────────┤
│ Libraries: BLAS, cuBLAS, cuDNN (C, C++, CUDA, assembly)      │
├──────────────────────────────────────────────────────────────┤
│ Hardware: CPU (SIMD), GPU (CUDA cores, tensor cores), TPU    │
└──────────────────────────────────────────────────────────────┘
```

For each language or tool below: first what it is, then where it fits, its strengths and weaknesses, and why Python still sits on top.

## C and C++

**C** is the low-level compiled language used throughout this series (notebooks 01 and 03). **C++** extends C with tools for organising large programs (classes, templates, a large standard library) while keeping the same speed and control over memory.

| Aspect | Assessment |
|---|---|
| Role in AI | The layer underneath almost everything: the core of PyTorch and TensorFlow, and **inference engines** (programs that run already-trained models efficiently) such as `llama.cpp`, ONNX Runtime and TensorRT |
| Strengths | Full control of memory and hardware; mature compilers; runs everywhere, from servers to small embedded devices |
| Weaknesses | Slow to write, easy to get wrong (a memory mistake can crash the program or corrupt data silently), long compile times; poor for exploring ideas |
| Why Python stays on top | Nobody wants to recompile a program to try a different setting. C++ is where libraries are *built*, not where models are *designed* |

C++ matters most when a trained model is **deployed**: put to work in a phone, a car or a server answering millions of requests. There, Python is often left out entirely. `llama.cpp` runs large language models on ordinary laptops precisely because it is lean C/C++ with hand-tuned code.

## CUDA and its alternatives

**CUDA** is NVIDIA's platform for programming its GPUs (notebook 05): a C++ extension, a compiler and libraries such as cuBLAS and cuDNN.

| Aspect | Assessment |
|---|---|
| Role in AI | The reason NVIDIA dominates AI hardware: every major framework supports it first |
| Strengths | Almost twenty years of tools and libraries |
| Weaknesses | **Vendor lock-in**: CUDA code only runs on NVIDIA GPUs. Writing fast CUDA code requires deep expertise |
| Alternatives | AMD's ROCm/HIP (closely modelled on CUDA), Apple's Metal, Intel's oneAPI/SYCL |

Very few AI developers write CUDA directly: they call libraries that were written in it.

## Triton

**Triton** is a language embedded in Python for writing GPU code at the level of blocks of data rather than individual threads (deep dive [GPU performance](gpu_performance.ipynb)).

| Aspect | Assessment |
|---|---|
| Role in AI | Writing custom GPU operations without CUDA C |
| Strengths | Close to library speed with a fraction of the code; `torch.compile` (PyTorch's compiler) generates Triton code automatically; supports NVIDIA and AMD GPUs |
| Weaknesses | GPU-only and Linux-first; still requires understanding how data moves through memory |
| Why it matters | It brings GPU programming back into Python, which strengthens Python's position instead of competing with it |

## JAX and XLA

**JAX** is not a separate language. It is a Python library, created at Google, with a NumPy-like interface. Its functions can be **traced** (JAX records which operations they perform) and compiled by **XLA** (*Accelerated Linear Algebra*), a compiler that merges operations and generates fast code for CPUs, GPUs and **TPUs** (Google's AI chips).

```python
import jax
import jax.numpy as jnp

@jax.jit          # compile the whole function with XLA on the first call
def dense_layer(inputs, weights, bias):
    return jax.nn.relu(inputs @ weights + bias)
```

It follows the same pattern as Numba (deep dive [Compiling Python](compiling_python.ipynb)) and Triton: Python syntax, compiled execution. JAX is widely used in research, and it is the standard way to program TPUs.

## Rust

**Rust** is a compiled language as fast as C and C++, designed so that the compiler itself prevents most memory mistakes before the program runs.

| Aspect | Assessment |
|---|---|
| Role in AI | Tools and infrastructure rather than models: `uv` and `ruff` (tools for Python developers), Polars (fast data tables), Hugging Face's `tokenizers` (splitting text into tokens) and `safetensors` (storing model weights), servers that run models |
| Strengths | C-level speed with memory safety checked by the compiler; an excellent package manager (Cargo); easy to call from Python |
| Weaknesses | Hard to learn; a small ecosystem for numerical computing and deep learning |
| Why Python stays on top | Rust is increasingly used to build *tools for Python*, making Python better instead of replacing it |

This project is an example: `uv`, the tool that installed every dependency here, is written in Rust.

## Julia

**Julia** is a language designed for scientific computing. It looks like Python or MATLAB, but it is JIT-compiled (notebook 01) through LLVM, the same technology Numba uses, so ordinary loops run at C-like speed.

| Aspect | Assessment |
|---|---|
| Role in AI | Scientific computing and *scientific machine learning* (combining neural networks with physical equations), climate models, finance |
| Strengths | Solves the **two-language problem**: researchers prototype and optimise in the same language, instead of prototyping in Python and rewriting slow parts in C |
| Weaknesses | A much smaller community than Python's; a noticeable compilation delay when a session starts; fewer pre-trained models and production tools |
| Why Python stays on top | Mainstream deep learning (language models, computer vision) runs on PyTorch and JAX; Julia's speed advantage matters less when the heavy work happens inside libraries anyway |

```julia
# Julia: loops are fast without any decorator or type declaration
function matmul(a, b)
    n = size(a, 1)
    c = zeros(n, n)
    for j in 1:n, k in 1:n, i in 1:n   # column-major: i innermost is the cache-friendly order
        c[i, j] += a[i, k] * b[k, j]
    end
    return c
end
```

Julia stores matrices **column by column** (like Fortran and MATLAB), so the cache-friendly loop order is the opposite of C's. The lesson of the deep dive [Loop order and the cache](loop_order_and_cache.ipynb) still applies, mirrored.

## Mojo

**Mojo** is a new language created by Chris Lattner (the creator of LLVM and of Apple's Swift language) at the company Modular. Its syntax is close to Python's, but it adds static types (notebook 01) and is compiled, with direct access to SIMD and to GPUs.

| Aspect | Assessment |
|---|---|
| Role in AI | Writing high-performance code for AI that runs on CPUs and on GPUs from several vendors |
| Strengths | Python-like syntax with C-like speed; memory safety inspired by Rust; can call and be called from Python |
| Weaknesses | A very young ecosystem and community; most Python libraries cannot be used as native Mojo code |
| Status | It reached **version 1.0 in August 2026**, and its compiler was released as open source (Apache 2.0 licence) the same month |

Mojo competes mainly with CUDA C++ and Triton for writing the lower layers, rather than with Python at the top. Whether it becomes widely adopted is an open question worth following.

## Why Python is still on top

Ranked by raw speed, Python comes last. Ranked by use in AI, it comes first. The reasons:

- **Ecosystem**: PyTorch, JAX, scikit-learn, Hugging Face, pandas, Jupyter. The models, tutorials and answers are in Python.
- **Speed of iteration**: research means trying many ideas quickly, and an interpreted language with notebooks is the fastest way to do that (notebook 01).
- **The heavy work is not in Python**: as measured in notebooks 04 and 05, the interpreter's slowness stops mattering once the work is handed to BLAS, cuBLAS or compiled code.
- **Interoperability**: Python can call C, C++, Fortran, CUDA and Rust easily, so the fast languages join the stack *underneath* it instead of replacing it.

The fair criticism is that the **glue itself** can be slow. Preparing data, loading files and control logic written in Python can become bottlenecks. That is exactly where Numba, Cython, Rust extensions and new versions of CPython (which are removing the old limit that prevented Python threads from running at the same time) are improving things.

## Key ideas

- AI software is built in layers: Python at the top, compiled libraries and hardware underneath.
- C, C++, CUDA and Rust mostly *build* the lower layers; Triton, JAX and Numba bring compilation back into Python.
- Julia and Mojo try to be both easy and fast in a single language; they are promising, but their ecosystems are small.
- Python stays on top because of its ecosystem and speed of development, not its execution speed.

## Reference results

Measured with this project's notebooks on the author's machine ("Where" names the main notebook or the deep dive): an AMD Ryzen 9 9900X (12 cores) and an NVIDIA GeForce RTX 5060 Ti. They use CPython 3.14, NumPy 2.5 with OpenBLAS, and PyTorch 2.14 with CUDA 13. **Your numbers will differ**: compare the ratios, not the absolute values. (1 TFLOPS = 1,000 GFLOPS.)

| Implementation | Where | Precision | n | GFLOPS |
|---|---|---|---|---|
| Pure CPython, naive loop | 03 | float64 | 256 | 0.11 |
| Cython, no types | compiling_python | float64 | 256 | 0.17 |
| C `ijk` order, `-O3` | loop_order_and_cache | float64 | 1024 | 0.5 |
| PyPy, naive loop | compiling_python | float64 | 256 | 1.6 |
| C `ikj` order, `-O3` | 03 | float64 | 1024 | 36 |
| Numba / Cython with types, `ikj` | compiling_python | float64 | 256 | 34–40 |
| Numba, parallel | compiling_python | float64 | 1024 | 118 |
| NumPy (OpenBLAS), 1 thread | inside_blas | float64 | 4096 | 131 |
| GPU, PyTorch | gpu_performance | float64 | 4096 | 360 |
| NumPy (OpenBLAS), all threads | inside_blas | float64 | 4096 | about 900 |
| CPU, NumPy or PyTorch, all cores | 05 | float32 | 4096 | 1,400–1,800 |
| Naive CUDA kernel | 05 | float32 | 4096 | 1,500 |
| GPU, cuBLAS | 05 | float32 | 4096 | 17,100 |
| GPU, TF32 on tensor cores | gpu_performance | TF32 | 4096 | 24,800 |
| Triton kernel | gpu_performance | float16 | 4096 | 47,300 |
| GPU, float16 on tensor cores | 05 | float16 | 4096 | 49,100 |

From the first row to the last, the same mathematical operation runs about **500,000 times faster**, almost six orders of magnitude:

- **Compiling instead of interpreting**, with a cache-friendly loop, accounts for two to three of them: pure Python against C or Numba.
- **The rest** comes from tuned libraries, multithreading, specialised hardware and lower precision.

Python code can reach every one of those through NumPy, Numba or PyTorch, and that is the whole argument of this series.