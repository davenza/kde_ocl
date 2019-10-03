This repository implements Gaussian [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) using 
OpenCL to achieve important performance gains.


The Python interface is based on the [Scipy's `gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) class, 
so it should be pretty easy to replace the CPU implementation of `gaussian_kde` with the
OpenCL implementation in this repository `gaussian_kde_ocl`.


## Example Code


```python
import numpy as np
from kde_ocl import gaussian_kde_ocl

# Generate dummy training data (10000 instances of 2D data)
train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 10000)
# Generate dummy test data (10000 instances of 2D data)
test = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)

# Train the KDE model
kde = gaussian_kde_ocl(train)

# Get the pdf of each test point. This is equivalent to kde.pdf(test)
pdf = kde(test)

# Get the logpdf of each test point. This is equivalent to kde.pdf(test)
logpdf = kde.logpdf(test)
```

*The interface is mostly the same as Scipy's `gaussian_kde`, but the axis order is changed*. For example, training a 
Scipy's `gaussian_kde` with a numpy array of shape (10000, 2) is interpreted as two instances of 10000 dimensions. In
`gaussian_kde_ocl`, this data is interpreted as 10000 instances of 2 dimensions. This change makes easier to work with
`pandas` dataframes:

```python
import pandas as pd
import numpy as np
from kde_ocl import gaussian_kde_ocl

# Create pandas dataframe 
a = np.random.normal(0, 1, 5000)
b = np.random.normal(3.2, np.sqrt(1.8), 5000)
data = pd.DataFrame({'a': a, 'b': b})

# Train KDE model
kde = gaussian_kde_ocl(data.values)

# Evaluate one point
logpdf = kde.logpdf([1.1, 2.3])
```

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

### Python Dependencies

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
pytest --times
``

To run only the OpenCL benchmarks:

``
pytest --times-ocl
``

To run only the Scipy's `gaussian_kde` benchmarks:


``
pytest --times-scipy
``
