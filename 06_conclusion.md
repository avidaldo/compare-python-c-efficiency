# Conclusion: why Python rules AI

## The puzzle

Notebook 01 showed that Python is one of the slowest mainstream languages: an interpreted loop runs tens of times slower than the same loop compiled from C. Notebook 02 showed that AI is, above all, matrix multiplication, repeated trillions of times. So how can Python be the main language of AI?

## The answer: Python describes, compiled code computes

In practice, Python almost never performs the heavy computation of an AI program. It **describes** what to compute and **calls** libraries that do it, and those libraries are written in compiled languages and run on hardware designed for the job:

```text
┌──────────────────────────────────────────────────────────────┐
│ Python: notebooks, training scripts, data preparation        │  ← where AI developers work
├──────────────────────────────────────────────────────────────┤
│ Libraries: NumPy, pandas, scikit-learn, PyTorch, TensorFlow  │  ← Python interface, compiled core
├──────────────────────────────────────────────────────────────┤
│ Compiled engines: BLAS (C, Fortran), cuBLAS (CUDA), C++      │  ← expert-tuned code
├──────────────────────────────────────────────────────────────┤
│ Hardware: CPU cores, GPU with thousands of cores             │
└──────────────────────────────────────────────────────────────┘
```

This is what it means to call Python a **glue language**: it connects fast components written in other languages. The slowness of its interpreter costs a little on each call, but nothing on the billions of operations performed inside each call.

## The journey in numbers

The same matrix multiplication, measured in these notebooks on the author's computer (an AMD Ryzen 9 9900X processor with 12 cores and an NVIDIA GeForce RTX 5060 Ti graphics card). Your numbers will be different, but the proportions will be similar.

| Notebook | How the matrices were multiplied | GFLOPS | Compared with pure Python |
|---|---|---|---|
| 03 | Pure Python, three nested loops | 0.11 | 1× |
| 03 | C, the same three loops | 2.4 | about 20× |
| 03 | C, loops reordered to read memory in order | 35 | about 300× |
| 04 | NumPy (BLAS, all CPU cores) | 500 | about 5,000× |
| 05 | Our CUDA kernel on the GPU (32-bit numbers) | 1,500 | about 15,000× |
| 05 | cuBLAS on the GPU (32-bit numbers) | 17,000 | about 150,000× |
| 05 | cuBLAS on the GPU (16-bit numbers, tensor cores) | 49,000 | about 450,000× |

From the first row to the last, the same operation runs almost half a million times faster. And every row from NumPy downwards was started with a single line of Python.

## What happens in a real AI workflow

Consider one step of training a neural network with PyTorch. In Python, the programmer writes something like "take this batch of examples, pass it through the model, measure the error, adjust the weights". Each of those phrases is one or a few Python calls. Behind them, PyTorch's C++ core decides what to do, and cuBLAS and other CUDA libraries perform billions of multiplications on the GPU. The Python part of each step takes microseconds; the GPU part takes milliseconds. Python's slowness is simply not where the time goes.

The same is true outside deep learning: pandas runs its table operations in C, and scikit-learn trains its models in Cython, C and C++.

## Why this combination works so well

- **Each language does what it is best at.** Python is easy to write, read and try interactively, so people can experiment quickly. C, C++ and CUDA are fast, so the computations run at the speed of the hardware.
- **Experts optimise once, everyone benefits.** A handful of specialists tune BLAS and cuBLAS for each new processor, and every Python program that multiplies matrices becomes faster without changing a line.
- **The hardware keeps improving underneath.** When a new GPU arrives, the libraries are updated to use it, and Python code written years earlier runs on it unchanged.

## The limit: when the glue itself is slow

The combination breaks down when a program does its heavy work in Python loops instead of library calls: for example, processing a dataset value by value in a `for` loop. Then the interpreter is back in the path of the work. The fix is almost always the same: express the work as operations on whole arrays or tensors (vectorisation), or, when no library function fits, compile that part with a tool such as Numba.

## Where to go next

The deep dives explore the details behind each notebook:

- [How programs run](deep-dives/how_programs_run.ipynb): bytecode, type checking, JIT compilation and machine code.
- [Numbers and memory](deep-dives/numbers_and_memory.ipynb): floating-point numbers, how Python stores values, and the processor's cache.
- [Matrix multiplication in AI](deep-dives/matmul_in_ai.ipynb): image filters, attention and a real Transformer block measured with a profiler.
- [Loop order and the cache](deep-dives/loop_order_and_cache.ipynb): why reading memory in order matters so much.
- [Inside BLAS](deep-dives/inside_blas.ipynb): threads, cores and how an expert library organises its work.
- [Compiling Python](deep-dives/compiling_python.ipynb): Numba, Cython and PyPy.
- [GPU performance](deep-dives/gpu_performance.ipynb): timing GPU code, data transfers, precision formats and Triton.
- [Languages for AI beyond Python](deep-dives/language_landscape.md): C++, CUDA, Triton, JAX, Rust, Julia and Mojo.
