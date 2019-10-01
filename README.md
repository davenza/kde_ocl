This repository implements Gaussian [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) using 
OpenCL to achieve important performance gains:


The Python interface is based on the [Scipy's `gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) class, 
so it should be pretty easy to replace the CPU implementation of `gaussian_kde` with the
OpenCL implementation in this repository `gaussian_kde_ocl`.

## Performance

This is a comparison of the `gaussian_kde_ocl` and Scipy's `gaussian_kde` with 2D data and the following configuration:

- CPU: Intel i7 6700k.
- GPU: AMD RX 460.
- Python 3.7.3
- Ubuntu 16.04


Training instances / Test instances | `gaussian_kde_ocl.pdf()`  | `gaussian_kde.pdf()` | Speedup `pdf()` | `gaussian_kde_ocl.logpdf()` | `gaussian_kde.logpdf()` | Speedup `logpdf()` |
------------------------------------|---------------------------| ---------------------|-----------------|-----------------------------|-------------------------|--------------------|
100,000 / 1,000                     | 0.2177 s                  | 1.9567 s             | 9.65x           | 0.2522 s                    | 8.1299 s                | 32.23x             |
1,000 / 10,000,000                  | 19.2637 s                 | 365.7575 s           | 18.99x                | 36.6198 s                   | MemoryError             | NA                 |
100 / 10,000                        | 0.006212 s                | 0.018888 s           | 3.04x           | 0.008852 s                  | 0.037864 s              | 4.28x              |


## Dependencies

The library is Python 2/3 compatible. Currently, is tested in Ubuntu 16.04, but should be compatible with other operating systems where
there are OpenCL GPU support.

### Python dependencies

The project has the following Python dependencies:

``
cffi
numpy
six
``

You can install them with:

``
pip install cffi numpy six
``

### Rust

The [Rust](https://www.rust-lang.org/) compiler must be installed in the system. Check out [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) for more information.

### OpenCL

The GPU drivers that enable OpenCL should be installed.