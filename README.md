This repository implements Gaussian [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) using 
OpenCL to achieve important performance gains.


The Python interface is based on the [Scipy's `gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) class, 
so it should be pretty easy to replace the CPU implementation of `gaussian_kde` with the
OpenCL implementation in this repository `gaussian_kde_ocl`.

## Performance

This is a comparison of the `gaussian_kde_ocl` and Scipy's `gaussian_kde` with 2D data and the following configuration:

- CPU: Intel i7-6700K.
- GPU: AMD RX 460.
- Python 3.7.3
- Ubuntu 16.04


### ``pdf()`` method

Training instances / Test instances | `gaussian_kde_ocl.pdf()`    | `gaussian_kde.pdf()`            | Speedup |
------------------------------------|-----------------------------| --------------------------------|-----------------|
100,000 / 1,000                     | 218.6474 &plusmn; 1.5901 ms | 1,911.0764 &plusmn; 50.8762 ms  | 8.74x   |
1,000 / 10,000,000                  | 18.8643 &plusmn; 0.07322 s  | 237.3429 &plusmn; 1.1765 s      | 12.58x  |
100 / 10,000                        | 4.4533 &plusmn; 0.7297 ms   | 18.0684 &plusmn; 0.3302 ms      | 4.46x   |

### ``logpdf()`` method


Training instances / Test instances | `gaussian_kde_ocl.logpdf()` | `gaussian_kde.logpdf()`         | Speedup |
------------------------------------|-----------------------------|---------------------------------|---------|
100,000 / 1,000                     | 261.1466 &plusmn; 6.3932 ms | 6,798.4730 &plusmn; 420.2878 ms | 26.03x  |
1,000 / 10,000,000                  | 36.3143 &plusmn; 0.02916 s  | MemoryError                     | NA      |
100 / 10,000                        | 8.827 &plusmn; 0.7442 ms    | 34.1114 &plusmn; 1.3060 ms      | 3.86x   |


## Current Limitations

- Only C order (the default) numpy arrays can be used as traning/test datasets.
- Only Gaussian kernels are implemented.
- OpenCL device is selected automatically.

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


## Testing

Tests are run using pytest and requires `scipy` to compare `gaussian_kde_ocl` with Scipy's `gaussian_kde`. Install them:

``
pip pytest scipy
``

Run the tests with:

``
pytest
``

### Benchmarks

To run the benchmarks, `pytest-benchmark` is needed:

``
pip pytest-benchmark
``

Then, execute the tests with benchmarks enabled:

``
pytest --benchmark
``