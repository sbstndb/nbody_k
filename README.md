# Introduction

Here is a simple NBody simulator. It uses a naive algorithm - in a O(n²) complexity. 
The aim of this project is to have a first look at the Kokkos library.

From the [Kokkos](https://github.com/kokkos) github repo : 
> The Kokkos C++ Performance Portability Ecosystem is a production level solution for writing modern C++ applications in a hardware agnostic way.
Thanks to Kokkos, you can compile this code for multiple backends like OpenMP, CUDA, HIP. 


**Disclaimer:** Please note that this project is a work in progress and may contain errors or programming oversights due to its experimental nature. Your understanding and feedback are appreciated as we continue to develop and refine this code.


# Prerequisites
The code use Kokkos as a performance portability library. Then, you must have it to compile the project.
I suggest you to install it and read the documentation for further understanding. 


# Compilation
Compile the code by specifying your kokkos install directory with the following commands : 
```
cmake -B build -S . -DKokkos_ROOT=/usr/local/lib64/cmake/Kokkos
cmake --build build/
```
I Suggest you to read the Kokkos documentation if you want to compile with a specific backend. 


# Usage
You can currently launch the SOA version named `NBody_SOA` with :
```
./build/NBody_SOA -N 10240 -nrepeat 100 -f 100 -d 0.001
```
With the following arguments : 
- `-n` particles
- `-nrepeat` iterations
- `d` timestep
- `f` frames - each frame consume `nrepeat` iterations and save the values in the `frames` directory

# Todo 
### To-Do List

- [ ] Refactoring
- [ ] Hierarchical parallelism with Kokkos
- [ ] Shared memory kernels with scratch pads
- [ ] Kokkos::SIMD usage
- [ ] Barnes-Hut implementation
- [ ] Fast Multipole Method (FMM) implementation
- [ ] Add periodic boundary conditions
- [ ] Python animation generator
- [x] Set up the CMake build system
- [x] Test the code on different backends (CUDA, OpenMP, Serial)
