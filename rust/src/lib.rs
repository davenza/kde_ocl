//! The crate `kde_ocl_sys` implements
//! [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) (KDE) using
//! OpenCL to accelerate the computation with GPUs. Currently, it only implements the (multivariate)
//! Gaussian kernel. This crate is an auxiliary library for the Python code using it along with this
//! code. For this reason, it receives Numpy arrays as input, and writes the results also in Numpy
//! arrays.
//!
//! The equation for the KDE with $`n`$ training points of dimensionality $`d`$ is the following
//! expression:
//!
//! ```math
//! \hat{f}_{n}(\mathbf{x}) = \frac{1}{n(2\pi)^{d/2}\sqrt{\lvert\mathbf{\Sigma}\rvert}}
//! \sum_{i=1}^{n} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}_{i})^{T}\Sigma^{-1}(\mathbf{x} -
//! \mathbf{x}_{i})\right)
//! ```
//! # Implementation
//!
//! The evaluation of each Gaussian is implemented using the Cholesky decomposition:
//!
//! ```math
//! \mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{T}
//! ```
//!
//! such that:
//!
//! ```math
//! (\mathbf{x} - \mathbf{x}_{i})^{T}(\mathbf{L}\mathbf{L}^T)^{-1}(\mathbf{x} - \mathbf{x}_{i}) =
//! (\mathbf{x} - \mathbf{x}_{i})^{T}\mathbf{L}^{-T}\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}) =
//! (\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}))^T(\mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i}))
//! ```
//! Let $`\mathbf{y}^{i} = \mathbf{L}^{-1}(\mathbf{x} - \mathbf{x}_{i})`$, $`\mathbf{y}^{i}`$ can be
//! easily obtained by forward-solving $`\mathbf{L}\mathbf{y}^{i} = (\mathbf{x} - \mathbf{x}_{i})`$
//! (quite easy to solve as $`\mathbf{L}`$ is lower triangular). Once $`\mathbf{y}^{i}`$ is solved:
//!
//! ```math
//! \left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}_{i})^{T}\Sigma^{-1}(\mathbf{x} - \mathbf{x}_{i})
//! \right) = -\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2}
//! ```
//!
//! Once this is computed, we only have to substract the constant (because it does not change for
//! every training instance in the KDE) quantity:
//! ```math
//!  \log n + \frac{d}{2}\log2\pi+ \log\lvert\mathbf{L}\rvert
//! ```
//! To obtain the log probability density function (logpdf) of a test point $`\mathbf{x}`$ with
//! respect to a training point $`\mathbf{x}_{i}`$, $`l_{i}`$:
//!
//! ```math
//! l_{i} = -\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n - \frac{d}{2}\log2\pi -
//! \log\lvert\mathbf{L}\rvert
//! ```
//! The pdf of a test point $`\mathbf{x}`$ with respect to a training point
//! $`\mathbf{x}_{i}`$ is equal to the exponentiation of the previous logpdf:
//!
//! ```math
//! \exp\left(-\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n - \frac{d}{2}\log2\pi -
//! \log\lvert\mathbf{L}\rvert\right)
//! ```
//!
//! To obtain the pdf of the KDE model, we have to sum over all the training points:
//!
//! ```math
//! \sum_{i=1}^{n}\exp\left(-\frac{1}{2} \sum_{j=1}^{d} (y_{j}^{i})^{2} - \log n -
//! \frac{d}{2}\log2\pi - \log\lvert\mathbf{L}\rvert\right)
//! ```
//! ## LogSumExp trick
//! For computing the logpdf of the KDE model, the
//! [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) is used for improved precision:
//!
//! ```math
//! \log\sum_{i}^{n} \exp x_{i} = \max_{i} x_{i} + \log\sum_{i}^{n} \exp\left(x_{i} -
//! \max_{i} x_{i}\right)
//! ```
//!
//! Then, the logpdf of the KDE model is:
//!
//! ```math
//! \max_{i} l_{i} + \log\sum_{i=1}^{n}\exp\left(l_{i} - \max_{i} l_{i}\right)
//! ```

extern crate libc;
extern crate ndarray;
extern crate num;

extern crate ocl;

use libc::{c_double, size_t};
use ndarray::{Array2, ShapeBuilder};
use ocl::{
    enums::{DeviceInfo, DeviceInfoResult},
    Buffer, ProQue,
};
use std::f64;
use std::mem;
use std::ptr;
use std::slice;

mod open_cl_code;

#[macro_use]
mod util_macros;

/// This struct represents a double Numpy array
#[repr(C)]
pub struct DoubleNumpyArray {
    ptr: *mut c_double,
    size: size_t,
    ndim: size_t,
    shape: *mut size_t,
    strides: *mut size_t,
}

/// Loads the meta-data of a Numpy array: its data pointer, and its shape and strides.
fn load_numpy_metadata<'a>(array: *const DoubleNumpyArray) -> (Vec<f64>, &'a [usize], &'a [usize]) {
    let array_vec = unsafe { slice::from_raw_parts((*array).ptr, (*array).size).to_vec() };

    let shape = unsafe { slice::from_raw_parts((*array).shape, (*array).ndim) };

    let strides = unsafe { slice::from_raw_parts((*array).strides, (*array).ndim) };

    (array_vec, shape, strides)
}

/// Loads a  2D Numpy array as a Rust's ndarray array. This function creates an
/// [Array2](https://docs.rs/ndarray/*/ndarray/type.Array2.html) instead of
/// an [ArrayD](https://docs.rs/ndarray/*/ndarray/type.ArrayD.html).
fn load_numpy_2d(array: *const DoubleNumpyArray) -> Array2<f64> {
    let (array_vec, shape, strides) = load_numpy_metadata(array);
    let mut arr_shape = [0; 2];
    arr_shape.copy_from_slice(shape);
    let mut arr_strides = [0; 2];
    arr_strides.copy_from_slice(strides);
    // We need to convert the strides from bytes to the number of elements.
    arr_strides
        .iter_mut()
        .for_each(|s| *s = *s / mem::size_of::<f64>());
    unsafe { Array2::from_shape_vec_unchecked(arr_shape.strides(arr_strides), array_vec) }
}

/// This enum represents the posible errors that can arise during the execution of the KDE.
/// The error codes can be used by Python to give an adequate response.
#[repr(C)]
#[derive(PartialEq)]
pub enum Error {
    NoError = 0,
    MemoryError,
}

/// Implements a KDE density model in OpenCL.
#[derive(Debug)]
pub struct GaussianKDE {
    /// Number of train instances.
    n: usize,
    /// Dimensionality of the training data.
    d: usize,
    /// Buffer containing the training data.
    training_data: Buffer<f64>,
    /// Buffer containing the Cholesky decomposition of the covariance matrix of the KDE.
    chol_cov: Buffer<f64>,
    /// Log of the normalization factor of the Gaussian.
    /// That is:
    /// ```math
    /// \log (2\pi)^{d/2} \sqrt{\lvert\mathbf{\Sigma}}\rvert
    /// ```
    lognorm_factor: f64,
}

/// Gets the maximum work group size of the default device. This is the preferred local work size
/// for many operations (especially reductions, such as sums or finding maximums).
fn get_max_work_size(pro_que: &ProQue) -> usize {
    match pro_que
        .device()
        .info(DeviceInfo::MaxWorkGroupSize)
        .expect("The maximum local work size could not be detected.")
    {
        DeviceInfoResult::MaxWorkGroupSize(s) => s,
        _ => unreachable!(),
    }
}

/// Fills a given OpenCL buffer with a value.
fn buffer_fill_value(pro_que: &ProQue, b: &Buffer<f64>, size: usize, value: f64) {
    let kernel_zeros = pro_que
        .kernel_builder("fill_value")
        .global_work_size(size)
        .arg(b)
        .arg(value)
        .build()
        .expect("Final result initialization failed.");
    unsafe {
        kernel_zeros.enq().expect("Error while filling the buffer.");
    }
}

/// Computes the lognorm factor: See [KDEDensityOcl](struct.KDEDensityOcl.html)
fn lognorm_factor(n: usize, d: usize, chol_cov: &Array2<f64>) -> f64 {
    let norm_factor = 0.5 * (d as f64) * (2.0 * f64::consts::PI).ln() + (n as f64).ln();
    // Summing the sqrt of the covariance determinant using the cholesky matrix.
    norm_factor
        + chol_cov
            .diag()
            .fold(0., |accum: f64, v| accum + (*v as f64).ln())
}

#[no_mangle]
pub unsafe extern "C" fn new_proque() -> *mut ProQue {
    // TODO: The OpenCL code should be included in the code to make easier distribute the library.


    let pro_que = ProQue::builder()
        .src(open_cl_code::OPEN_CL_CODE)
        .build()
        .expect("Error while creating OpenCL ProQue.");

    let proque_box = Box::new(pro_que);
    Box::into_raw(proque_box)
}

/// Initializes a `KDEDensityOcl`. It expects two `DoubleNumpyArray` with the cholesky decomposition
/// of the covariance matrix and the training data. The training data is expected to have shape
/// ($`n`$, $`d`$), where n is the number of instances and $`d`$ is the number of atributes. The
/// covariance matrix should have shape ($`d`$,$`d`$).
///
/// # Safety
///
/// This function is unsafe because it receives some Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data.
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_init(
    pro_que: *mut ProQue,
    chol: *const DoubleNumpyArray,
    training_data: *const DoubleNumpyArray,
    error: *mut Error,
) -> *mut GaussianKDE {
    let n = *(*training_data).shape;
    let d = *(*chol).shape;
    let chol_cov = load_numpy_2d(chol);
    let lognorm_factor = lognorm_factor(n, d, &chol_cov);
    let chol_vec = chol_cov.into_raw_vec();

    let pro_que = Box::from_raw(pro_que);

    let training_slice = slice::from_raw_parts((*training_data).ptr, (*training_data).size);

    let (training_buffer, chol_buffer) =
        copy_buffers!(pro_que, error, training_slice, chol_vec => ptr::null_mut());

    let kde = Box::new(GaussianKDE {
        n,
        d,
        training_data: training_buffer,
        chol_cov: chol_buffer,
        lognorm_factor,
    });

    let ptr_kde = Box::into_raw(kde);

    Box::into_raw(pro_que);

    *error = Error::NoError;
    ptr_kde
}

/// Frees the `KDEDensityOcl`.
#[no_mangle]
pub extern "C" fn gaussian_kde_free(kde: *mut GaussianKDE) {
    if kde.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(kde);
    }
}

/// Frees the `ProQue`.
#[no_mangle]
pub extern "C" fn gaussian_proque_free(pro_que: *mut ProQue) {
    if pro_que.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(pro_que);
    }
}

/// Computes the probability density function (pdf) evaluation of $`m`$ points given a KDE model.
/// The $`m`$ testing points are in the `testing_data` `DoubleNumpyArray` with shape ($`m`$, $`d`$).
/// The result is saved in the `result` array, that should have at least length $`m`$.
///
/// # Safety
///
/// This function is unsafe because it receives a Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data. Also, the kde and
/// result pointers should not point to NULL.
///
/// # Implementation
///
/// To compute the pdf, it iterates over the training data or the test data depending on which data
/// set has more instances. The general procedure is discussed in the
/// [main description](index.html).
///
/// ## Iterating over the test data
///
/// If $`\mathbf{D}`$ is the $`n \times d`$ matrix containing the training instances, and
/// $`\mathbf{t}^{k}`$ is a test instance. We iterate over all the $`\mathbf{t}^{k},\;
/// k=1,\ldots,m`$
///
/// ### Substract kernel
///
/// The substract OpenCL kernel, substracts the test instance from
/// all the training data:
///
/// ```math
/// \mathbf{D} = \begin{bmatrix}
/// d_{11} & \cdots & d_{1d}\\
/// \vdots & \ddots & \vdots\\
/// d_{n1} & \cdots & d_{nd}\\
/// \end{bmatrix},
/// \mathbf{t}^{k} = \begin{bmatrix} t_{1}^{k} & \cdots & t_{d}^{k}\end{bmatrix},\;
/// \text{substract}(\mathbf{D}, \mathbf{t}) = \mathbf{S}^{k} = \begin{bmatrix}
/// d_{11} - t_{1}^{k} & \cdots & d_{1d} - t_{d}^{k}\\
/// \vdots & \ddots & \vdots\\
/// d_{n1} - t_{1}^{k} & \cdots & d_{nd} - t_{d}^{k}\\
/// \end{bmatrix}
/// ```
///
/// ### Solve kernel
///
/// The solve kernel performs forward-solving over the substracted matrix. If $`\mathbf{L}`$ is the:
/// Cholesky matrix, and $`\mathbf{S}`$ the substracted matrix, the solved matrix $`\mathbf{V}`$
/// should hold:
///
/// ```math
///  \mathbf{L}\mathbf{V}^{k} = \mathbf{S}^{k}
/// ```
/// ```math
///  \begin{bmatrix}
/// l_{11} & \mathbf{0} & 0\\
/// \vdots & l_{ii} & \mathbf{0}\\
/// l_{d1} & \cdots & l_{dd}\\
///  \end{bmatrix}
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix} =
/// \begin{bmatrix}
/// s_{11} & \cdots & s_{1n}\\
/// \vdots & \ddots & \vdots\\
/// s_{d1} & \cdots & s_{dn}\\
/// \end{bmatrix}
/// ```
/// Then,
///
/// ```math
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix} =
/// \begin{bmatrix}
/// s_{11} / l_{11} & \cdots & v_{1n} / l_{11}\\
/// \vdots & \ddots & \vdots\\
/// \frac{s_{d1} - \sum_{j=1}^{d-1} v_{j1}l_{dj}}{l_{dd}} & \cdots &
/// \frac{s_{dn} - \sum_{j=1}^{d-1} v_{jn}l_{dj}}{l_{dd}}\\
/// \end{bmatrix} =
///
/// \begin{bmatrix} (\mathbf{y}^{1})^{T} & \cdots (\mathbf{y}^{i})^{T} &
/// \cdots (\mathbf{y}^{n})^{T}\end{bmatrix}
/// ```
///
/// ### Square kernel
///
/// The square kernel squares every element of a matrix. So,
/// ```math
/// \begin{bmatrix}
/// v_{11} & \cdots & v_{1n}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1} & \cdots & v_{dn}\\
/// \end{bmatrix},\; \text{square}(\mathbf{V}) = \mathbf{W}^{k} =
/// \begin{bmatrix}
/// v_{11}^{2} & \cdots & v_{1n}^{2}\\
/// \vdots & \ddots & \vdots\\
/// v_{d1}^{2} & \cdots & v_{dn}^{2}\\
/// \end{bmatrix}
/// ```
///
/// ### Sumout kernel
///
/// It sums over all the rows in the $`\mathbf{V}`$ matrix and exponentiates the sum.
///
/// ```math
/// \mathbf{W} = \begin{bmatrix}
/// w_{11} & \cdots & w_{1n}\\
/// \vdots & \ddots & \vdots\\
/// w_{d1} & \cdots & w_{dn}\\
/// \end{bmatrix},\;
/// \text{sumout}(\mathbf{W}) =
/// \mathbf{u}^{k} = \begin{bmatrix}
/// \exp\left(\sum_{i=1}^{d} w_{i1}\right) & \cdots & \exp\left(\sum_{i=1}^{d} w_{in}\right)\\
/// \end{bmatrix}
/// ```
///
/// ### sum_gpu function
///
/// It sums all the elements in a vector. The sum of the vector $`\mathbf{u}`$ is the pdf for the
/// instance $`\mathbf{t}`$:
///
/// ```math
/// \hat{f}_{n}(\mathbf{t}) = \sum_{i=1}^{n} u_{i}^{k}
/// ```
/// ## Iterating over the train data
///
/// When iterating over the train data, the kernels `substract`, `solve`, `square` and `sumout` are
/// applied exactly as in [Iterating over the test data](#iterating-over-the-test-data), but in
/// this case the test data and the train instance $`\mathbf{r}^{k},\; k=1,\ldots,n`$
/// is substracted.
///
/// Then the vector $`\mathbf{u}^{k}`$ represents the contribution of the $`\mathbf{r}^{k}`$
/// instance to every other test instance.
///
/// ### Sum_vectors kernel
///
/// The pdf result for a test instance implies the sum over all the $`\mathbf{u}^{k}`$ vectors:
///
/// ```math
///     \hat{f}_{n}(\mathbf{t}^{j}) = \sum_{k=1}^{n} u_{i}^{j}
/// ```
///
/// These sums are performed all at once using the sum_vectors kernel, that sums two vectors. If
/// $`\mathbf{v} = \begin{bmatrix} v_{1} & \cdots & v_{n}\end{bmatrix}`$ and
/// $`\mathbf{w} = \begin{bmatrix} w_{1} & \cdots & w_{n}\end{bmatrix}`$, then:
///
/// ```math
///     \text{sum\_vectors}(\mathbf{v}, \mathbf{w}) = \begin{bmatrix} v_{1} + w_{1} & \cdots &
///         v_{n} + w_{n}\end{bmatrix}
/// ```
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_pdf(
    kde: *mut GaussianKDE,
    pro_que: *mut ProQue,
    testing_data: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);

    let m = *(*testing_data).shape;
    let final_result = slice::from_raw_parts_mut(result, m);

    let max_work_size = get_max_work_size(&pro_que);

    let d = kde_box.d;
    //    Iterates over training or testing data?
    let len_iteration = if kde_box.n >= m { kde_box.n } else { m };

    pro_que.set_dims(len_iteration * d);

    let local_size = if len_iteration < max_work_size {
        len_iteration
    } else {
        max_work_size
    };
    let num_groups = (len_iteration as f32 / local_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*testing_data).ptr, m * d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        empty_buffers!(pro_que, error, f64, m, len_iteration * d, len_iteration);

    let kernel_substract = {
        let (matrix_buf, vec_buf) = if kde_box.n >= m {
            (&kde_box.training_data, &test_instances_buffer)
        } else {
            (&test_instances_buffer, &kde_box.training_data)
        };

        pro_que
            .kernel_builder("substract")
            .arg(matrix_buf)
            .arg(vec_buf)
            .arg(&tmp_matrix_buffer)
            .arg_named("row", &0u32)
            .arg(d as u32)
            .build()
            .expect("Kernel substract build failed.")
    };

    let kernel_solve = pro_que
        .kernel_builder("solve")
        .global_work_size(len_iteration)
        .arg(&tmp_matrix_buffer)
        .arg(&kde_box.chol_cov)
        .arg(d as u32)
        .build()
        .expect("Kernel solve build failed.");

    let kernel_square = pro_que
        .kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build()
        .expect("Kernel square build failed.");

    let kernel_sumout = pro_que
        .kernel_builder("sumout")
        .global_work_size(len_iteration)
        .arg(&tmp_matrix_buffer)
        .arg(&tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde_box.lognorm_factor)
        .build()
        .expect("Kernel sumout build failed.");

    if kde_box.n >= m {
        for i in 0..m {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract
                .enq()
                .expect("Error while executing substract kernel.");
            kernel_solve
                .enq()
                .expect("Error while executing solve kernel.");
            kernel_square
                .enq()
                .expect("Error while executing square kernel.");
            kernel_sumout
                .enq()
                .expect("Error while executing sumout kernel.");
            sum_gpu_vec(
                &pro_que,
                &tmp_vec_buffer,
                len_iteration,
                max_work_size,
                local_size,
                num_groups,
            );

            tmp_vec_buffer
                .copy(&final_result_buffer, Some(i), Some(1))
                .queue(pro_que.queue())
                .enq()
                .expect("Error copying to result buffer.");
        }
    } else {
        buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

        for i in 0..kde_box.n {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract
                .enq()
                .expect("Error while executing substract kernel.");
            kernel_solve
                .enq()
                .expect("Error while executing solve kernel.");
            kernel_square
                .enq()
                .expect("Error while executing square kernel.");
            kernel_sumout
                .enq()
                .expect("Error while executing sumout kernel.");

            let kernel_sumvectors = pro_que
                .kernel_builder("sum_vectors")
                .global_work_size(m)
                .arg(&final_result_buffer)
                .arg(&tmp_vec_buffer)
                .build()
                .expect("Kernel sum_vectors build failed.");

            kernel_sumvectors
                .enq()
                .expect("Error while executing sum_vectors kernel.");
        }
    }

    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
    *error = Error::NoError;
    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}

/// Sums all the elements in the vector buffer `sum_buffer` and places the result in the first
/// position of the `sum_buffer` (i.e., `sum_buffer[0]`). Keep in mind that the rest of the elements
/// of the buffer will be modified, so **it invalidates the rest of the data in the buffer**.
///
/// `global_size` is the length of the `sum_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// So, if `sum_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
///
/// After calling `sum_gpu_vec`, `sum_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \sum_{i=1}^{n} a_{i} & \ldots & \text{unexpected values} & \ldots
/// \end{bmatrix}
/// ```
fn sum_gpu_vec(
    pro_que: &ProQue,
    sum_buffer: &Buffer<f64>,
    mut global_size: usize,
    max_work_size: usize,
    mut local_size: usize,
    mut num_groups: usize,
) {
    while global_size > 1 {
        let kernel_sum_gpu = pro_que
            .kernel_builder("sum_gpu_vec")
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(sum_buffer)
            .arg_local::<f64>(local_size)
            .build()
            .expect("Kernel sum_gpu_vec build failed.");

        unsafe {
            kernel_sum_gpu
                .enq()
                .expect("Error while executing sum_gpu_vec kernel.");
        }

        global_size = num_groups;
        local_size = if global_size < max_work_size {
            global_size
        } else {
            max_work_size
        };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}

/// Computes the logarithm of the probability density function (log pdf) evaluation of $`m`$ points
/// given a KDE model. The $`m`$ testing points are in the `testing_data` `DoubleNumpyArray` with
/// shape ($`m`$, $`d`$). The result is saved in the `result` array, that should have at least
/// length $`m`$.
///
/// The [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) is used instead of applying a
/// logarithm to the results of `KDEDensity_logpdf_multiple_ocl` to obtain better precision.
///
/// # Safety
///
/// This function is unsafe because it receives a Numpy array pointers wrapped in the
/// `DoubleNumpyArray` struct. Those matrices should not point to invalid data. Also, the kde and
/// result pointers should not point to NULL.
///
/// # Implementation
///
/// To compute the pdf, it iterates over the training data or the test data depending on which data
/// set has more instances. The general procedure is discussed in the
/// [main description](index.html).
///
/// ## Iterating over the test data
///
/// If $`\mathbf{D}`$ is the $`n \times d`$ matrix containing the training instances, and
/// $`\mathbf{t}^{k}`$ is a test instance. We iterate over all the $`\mathbf{t}^{k},\;
/// k=1,\ldots,m`$
///
/// The first steps are as in [gaussian_kde_pdf](fn.gaussian_kde_pdf.html), applying the kernels
/// `substract`, `solve` and `square` exactly the same. However, instead of executing the `sumout`
/// kernel, the `logsumout` kernel is applied.
///
/// ### Logsumout kernel
///
/// It sums over all the rows in the $`\mathbf{V}`$ matrix without exponentiating the sum.
///
/// ```math
/// \mathbf{W} = \begin{bmatrix}
/// w_{11} & \cdots & w_{1n}\\
/// \vdots & \ddots & \vdots\\
/// w_{d1} & \cdots & w_{dn}\\
/// \end{bmatrix},\;
/// \text{logsumout}(\mathbf{W}) =
/// \mathbf{u}^{k} = \begin{bmatrix}
/// \sum_{i=1}^{d} w_{i1} & \cdots & \sum_{i=1}^{d} w_{in}\\
/// \end{bmatrix}
/// ```
///
/// ### max_gpu_vec_copy function
///
/// The `max_gpu_vec_copy` finds the maximum of the $`\mathbf{u}^{k}`$ vector.
/// [Its documentation](fn.max_gpu_vec_copy.html) contains more details about this function.
///
/// ```math
///     \text{max\_gpu\_vec\_copy}(\mathbf{u}^{k}) = \text{maxexp}= \max_{i} u_{i}^{k}
/// ```
/// The max value in the $`\mathbf{u}^{k}`$ vector is necessary to apply the
/// [LogSumExp trick](index.html#logsumexp-trick).
///
/// ### log_sum_gpu_vec function
///
/// The `log_sum_gpu_vec` sums the exponentiation of every element in the vector $`\mathbf{u}^{k}`$
/// substracting its maximum value. [Its documentation](fn.log_sum_gpu_vec.html) contains more
/// details about this function.
///
/// ```math
///     \text{log\_sum\_gpu\_vec}(\mathbf{u}^{k}) = \sum_{i=1}^{m}\exp\left(u_{i}^{k} -
///     \text{maxexp}\right)
/// ```
/// The substraction before the exponentiation is necessary to apply the
/// [LogSumExp trick](index.html#logsumexp-trick).
///
/// ### copy_log_pdf_result
///
/// It exponentiates the result of the `log_sum_gpu_vec` function and sums `maxexp`, to obtain the
/// log pdf for $`\mathbf{t}^{k}`$
///
/// ## Iterating over the train data
///
/// When iterating over the train data, there are two modes of behaviour depending on the available
/// memory in the OpenCL device as described in the
/// [log_pdf_iterate_train](fn.logpdf_iterate_train.html) documentation.
///
///
/// ### Low memory mode
///
/// If there is no enough memory to hold the logpdf for every pair of train and test instances, it
/// iterates twice over the train data.
///
/// In the first iteration, it finds the `maxexp` using a
/// different version of the `logsumout` kernel, that also saves the `maxexp` for each test
/// instance, called `logsumout_checkmax`. The previous steps are as in the
/// [Iterating over the test data](fn.gaussian_kde_logpdf.html#iterating-over-the-test-data) (i.e.,
/// applying the `substract`, `solve` and `square` kernels)
///
/// In the second iteration, it uses the `maxexp` to apply the
/// [LogSumExp trick](index.html#logsumexp-trick) with the `exp_and_sum` and `log_and_sum` kernels.
///
/// ### High memory mode
///
/// In this mode, the logpdf of each train and test instance pair is saved in a $`m \times n`$
/// matrix. This matrix is filled using the `substract`, `solve` and `square` while iterating over
/// all the train data and a new version of the
/// [logsumout kernel](fn.gaussian_kde_logpdf.html#logsumout-kernel) that works in the matrix called
/// `logsumout_to_matrix`.
///
/// Once the matrix is filled, the function `max_gpu_mat` finds the maximum element over each row
/// of the matrix to find the `maxexp` of each test instance. Then, the `exp_and_sum_mat` kernel is
/// executed (a modified kernel of the `exp_and_sum` kernel designed to work in matrices by making
/// the sum over every row after exponentiating the logpdf substracted with `maxexp`). Finally, the
/// `sum_gpu_mat` sums every exponentiated logpdf and the kernel `log_and_sum_mat` makes the
/// logarithm of the previous step result and sums `maxexp`.
#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_logpdf(
    kde: *mut GaussianKDE,
    pro_que: *mut ProQue,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let mut kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;
    if kde_box.n >= m {
        logpdf_iterate_test(&mut kde_box, &mut pro_que, x, result, error);
    } else {
        logpdf_iterate_train(&mut kde_box, &mut pro_que, x, result, error);
    }

    *error = Error::NoError;
    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}

/// We iterate over the test points if there are more training points.
unsafe fn logpdf_iterate_test(
    kde: &mut Box<GaussianKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let m = *(*x).shape;
    let d = kde.d;
    let final_result = slice::from_raw_parts_mut(result, m);
    let max_work_size = get_max_work_size(&pro_que);

    let n = kde.n;
    pro_que.set_dims(n * d);

    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (max_buffer, final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        empty_buffers!(pro_que, error, f64, num_groups, m, n * d, n);

    let kernel_substract = pro_que
        .kernel_builder("substract")
        .arg(&kde.training_data)
        .arg(&test_instances_buffer)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build()
        .expect("Kernel substract build failed.");

    let kernel_solve = pro_que
        .kernel_builder("solve")
        .global_work_size(n)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build()
        .expect("Kernel solve build failed.");

    let kernel_square = pro_que
        .kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build()
        .expect("Kernel square build failed.");

    let kernel_sumout = pro_que
        .kernel_builder("logsumout")
        .global_work_size(n)
        .arg(&tmp_matrix_buffer)
        .arg(&tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build()
        .expect("Kernel logsumout build failed.");

    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract kernel.");
        kernel_solve
            .enq()
            .expect("Error while executing solve kernel.");
        kernel_square
            .enq()
            .expect("Error while executing square kernel.");
        kernel_sumout
            .enq()
            .expect("Error while executing logsumout kernel.");

        max_gpu_vec_copy(
            &pro_que,
            &tmp_vec_buffer,
            &max_buffer,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        log_sum_gpu_vec(
            &pro_que,
            &tmp_vec_buffer,
            &max_buffer,
            n,
            max_work_size,
            local_work_size,
            num_groups,
        );

        let kernel_log_sum_gpu = pro_que
            .kernel_builder("copy_logpdf_result")
            .global_work_size(1)
            .arg(&tmp_vec_buffer)
            .arg(&max_buffer)
            .arg(&final_result_buffer)
            .arg(i as u32)
            .build()
            .expect("Kernel copy_logpdf_result build failed.");

        kernel_log_sum_gpu
            .enq()
            .expect("Error while executing copy_logpdf_result kernel.");
    }
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
}

/// Finds the maximum element in the vector buffer `max_buffer` and places the result in the first
/// position of `result_buffer` (i.e., `result_buffer[0]`). **This operation invalidates the rest
/// of the data in `result_buffer`**, but keeps constant `max_buffer`.
///
/// `global_size` is the length of the `sum_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// So, if `max_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
///
/// After calling `max_gpu_vec_copy`, `max_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} a_{1} & \ldots & a_{n}\end{bmatrix}
/// ```
/// and `result_buffer`:
///
/// ```math
///     \begin{bmatrix} \max_{i} a_{i} & \ldots & \text{unexpected values} & \ldots \end{bmatrix}
/// ```
fn max_gpu_vec_copy(
    pro_que: &ProQue,
    max_buffer: &Buffer<f64>,
    result_buffer: &Buffer<f64>,
    mut global_size: usize,
    max_work_size: usize,
    mut local_size: usize,
    mut num_groups: usize,
) {
    let kernel_max_gpu = pro_que
        .kernel_builder("max_gpu_vec_copy")
        .global_work_size(global_size)
        .local_work_size(local_size)
        .arg(max_buffer)
        .arg(result_buffer)
        .arg_local::<f64>(local_size)
        .build()
        .expect("Kernel max_gpu_vec_copy build failed.");

    unsafe {
        kernel_max_gpu
            .enq()
            .expect("Error while executing max_gpu_vec_copy kernel.");
    }

    global_size = num_groups;
    local_size = if global_size < max_work_size {
        global_size
    } else {
        max_work_size
    };
    num_groups = (global_size as f32 / local_size as f32).ceil() as usize;

    while global_size > 1 {
        let kernel_max_gpu = pro_que
            .kernel_builder("max_gpu_vec")
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(result_buffer)
            .arg_local::<f64>(local_size)
            .build()
            .expect("Kernel max_gpu_vec build failed.");

        unsafe {
            kernel_max_gpu
                .enq()
                .expect("Error while executing max_gpu_vec kernel.");
        }

        global_size = num_groups;
        local_size = if global_size < max_work_size {
            global_size
        } else {
            max_work_size
        };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}

/// Given a vector buffer `sum_buffer`:
///
/// ```math
///     \begin{bmatrix} s_{1} & \ldots & s_{n}\end{bmatrix}
/// ```
///
/// and a value located in the first position of `maxexp` (i.e., `maxexp[0]`):
///
/// Saves in the first position of `sum_buffer` (i.e. `sum_buffer[0]`), the following expression:
///
/// ```math
///     \sum_{i}^{n} \exp(s_{i} - \text{maxexp})
/// ```
///
/// `global_size` is the length of the `sum_buffer`. `max_work_size` is the maximum
/// number of work items in a work group for the selected device. `local_size` is the actual number
/// of work items in each work group. `num_groups` is the actual number of work groups.
///
/// This operation is a partial step to make to apply the
/// [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp). `maxexp[0]` should be the maximum
/// of all the elements in `sum_buffer`. **This operation invalidates the rest of the data in
/// `sum_buffer`**.
fn log_sum_gpu_vec(
    pro_que: &ProQue,
    sum_buffer: &Buffer<f64>,
    maxexp: &Buffer<f64>,
    mut global_size: usize,
    max_work_size: usize,
    mut local_size: usize,
    mut num_groups: usize,
) {
    let kernel_log_sum_gpu = pro_que
        .kernel_builder("log_sum_gpu_vec")
        .global_work_size(global_size)
        .local_work_size(local_size)
        .arg(sum_buffer)
        .arg_local::<f64>(local_size)
        .arg(maxexp)
        .build()
        .expect("Kernel log_sum_gpu_vec build failed.");

    unsafe {
        kernel_log_sum_gpu
            .enq()
            .expect("Error while executing log_sum_gpu_vec kernel.");
    }

    global_size = num_groups;
    local_size = if global_size < max_work_size {
        global_size
    } else {
        max_work_size
    };
    num_groups = (global_size as f32 / local_size as f32).ceil() as usize;

    while global_size > 1 {
        let kernel_sum_gpu_vec = pro_que
            .kernel_builder("sum_gpu_vec")
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(sum_buffer)
            .arg_local::<f64>(local_size)
            .build()
            .expect("Kernel sum_gpu_vec build failed.");

        unsafe {
            kernel_sum_gpu_vec
                .enq()
                .expect("Error while executing sum_gpu_vec kernel.");
        }

        global_size = num_groups;
        local_size = if global_size < max_work_size {
            global_size
        } else {
            max_work_size
        };
        num_groups = (global_size as f32 / local_size as f32).ceil() as usize;
    }
}

/// Iterates over the training data because there are less training points. There are two approaches
/// for iterating over the training data:
///
/// * The faster approach computes the log-likelihood for each training and test pair points in a
/// $`m \times n`$ matrix. Then, apply the
/// [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp) on each row of the matrix. This is
/// implemented in [logpdf_iterate_train_high_memory](fn.logpdf_iterate_train_high_memory.html).
/// * However, a $`m \times n`$ matrix can be too much large for the OpenCL device. The alternative
/// is to iterate twice along the training data. In the first iteration, the maximum
/// log-likelihood for each test point is stored in a $`m`$ vector buffer. In the second pass along
/// the train data, the logpdf can be computed using the LogSumExp trick with the pre-computed
/// maximums. This is implemented in
/// [logpdf_iterate_train_low_memory](fn.logpdf_iterate_train_low_memory.html).
unsafe fn logpdf_iterate_train(
    kde: &mut Box<GaussianKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    error: *mut Error,
) {
    let m = *(*x).shape;
    let tmp_vec_buffer = Buffer::<f64>::builder()
        .context(pro_que.context())
        .len(m * kde.n)
        .build();

    match tmp_vec_buffer {
        Ok(b) => logpdf_iterate_train_high_memory(kde, pro_que, x, result, &b, error),
        Err(_) => {
            let (tmp_vec_buffer,) = empty_buffers!(pro_que, error, f64, m);
            // TODO: If n < 2m, is it better to iterate over the training data?
            logpdf_iterate_train_low_memory(kde, pro_que, x, result, &tmp_vec_buffer, error);
        }
    }
}

/// Iterates twice over the training data to compute the logpdf of each test point using a $`m`$
/// vector as described in [logpdf_iterate_train](fn.logpdf_iterate_train.html).
unsafe fn logpdf_iterate_train_low_memory(
    kde: &mut Box<GaussianKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    tmp_vec_buffer: &Buffer<f64>,
    error: *mut Error,
) {
    let m = *(*x).shape;
    let d = kde.d;

    let final_result = slice::from_raw_parts_mut(result, m);

    let n = kde.n;
    pro_que.set_dims(m * d);

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        empty_buffers!(pro_que, error, f64, m, m, m * d);

    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

    let kernel_substract = pro_que
        .kernel_builder("substract")
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build()
        .expect("Kernel substract build failed.");

    let kernel_solve = pro_que
        .kernel_builder("solve")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build()
        .expect("Kernel solve build failed.");

    let kernel_square = pro_que
        .kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build()
        .expect("Kernel square build failed.");

    let kernel_sumout = pro_que
        .kernel_builder("logsumout")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build()
        .expect("Kernel logsumout build failed.");

    let kernel_sumout_checkmax = pro_que
        .kernel_builder("logsumout_checkmax")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build()
        .expect("Kernel logsumout_checkmax build failed.");

    let kernel_expsum = pro_que
        .kernel_builder("exp_and_sum")
        .global_work_size(m)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(&final_result_buffer)
        .build()
        .expect("Kernel exp_and_sum build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .build()
        .expect("Kernel log_and_sum build failed.");

    // Writes the max loglikelihoods in the max_buffer
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract kernel.");
        kernel_solve
            .enq()
            .expect("Error while executing solve kernel.");
        kernel_square
            .enq()
            .expect("Error while executing square kernel.");
        kernel_sumout_checkmax
            .enq()
            .expect("Error while executing logsumout_checkmax kernel.");
    }

    // Computes the loglikelihood using the max_buffer.
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract kernel.");
        kernel_solve
            .enq()
            .expect("Error while executing solve kernel.");
        kernel_square
            .enq()
            .expect("Error while executing square kernel.");
        kernel_sumout
            .enq()
            .expect("Error while executing logsumout kernel.");
        kernel_expsum
            .enq()
            .expect("Error while executing exp_and_sum kernel.");
    }
    kernel_log_and_sum
        .enq()
        .expect("Error while executing log_and_sum kernel.");
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
}

/// Iterates over the training data to compute the logpdf of each test point using a $`m \times n`$
/// matrix as described in [logpdf_iterate_train](fn.logpdf_iterate_train.html).
unsafe fn logpdf_iterate_train_high_memory(
    kde: &mut Box<GaussianKDE>,
    pro_que: &mut Box<ProQue>,
    x: *const DoubleNumpyArray,
    result: *mut c_double,
    tmp_vec_buffer: &Buffer<f64>,
    error: *mut Error,
) {
    let m = *(*x).shape;
    let d = kde.d;

    let final_result = slice::from_raw_parts_mut(result, m);

    let n = kde.n;
    pro_que.set_dims(m * d);

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m * d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);
    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        empty_buffers!(pro_que, error, f64, m * num_groups, m, m * d);

    let kernel_substract = pro_que
        .kernel_builder("substract")
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build()
        .expect("Kernel substract build failed.");

    let kernel_solve = pro_que
        .kernel_builder("solve")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build()
        .expect("Kernel solve build failed.");

    let kernel_square = pro_que
        .kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build()
        .expect("Kernel square build failed.");

    let kernel_sumout = pro_que
        .kernel_builder("logsumout_to_matrix")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(d as u32)
        .arg_named("sol_row", &0u32)
        .arg(n as u32)
        .arg(kde.lognorm_factor)
        .build()
        .expect("Kernel logsumout_to_matrix build failed.");

    let kernel_exp_and_sum = pro_que
        .kernel_builder("exp_and_sum_mat")
        .global_work_size((m, n))
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel exp_and_sum_mat build failed.");

    let kernel_log_and_sum = pro_que
        .kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build()
        .expect("Kernel log_and_sum_mat build failed.");

    // Writes the max loglikelihoods in the max_buffer
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract
            .enq()
            .expect("Error while executing substract kernel.");
        kernel_solve
            .enq()
            .expect("Error while executing solve kernel.");
        kernel_square
            .enq()
            .expect("Error while executing square kernel.");
        kernel_sumout.set_arg("sol_row", i as u32).unwrap();
        kernel_sumout
            .enq()
            .expect("Error while executing logsumout_to_matrix kernel.");
    }

    max_gpu_mat(
        &pro_que,
        tmp_vec_buffer,
        &max_buffer,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );
    kernel_exp_and_sum
        .enq()
        .expect("Error while executing exp_and_sum_mat kernel.");
    sum_gpu_mat(
        &pro_que,
        tmp_vec_buffer,
        m,
        n,
        max_work_size,
        local_work_size,
        num_groups,
    );

    kernel_log_and_sum
        .enq()
        .expect("Error while executing kernel log_and_sum_mat kernel.");
    final_result_buffer
        .cmd()
        .queue(pro_que.queue())
        .read(final_result)
        .enq()
        .expect("Error reading result data.");
}

/// Finds the maximum element of each row in the matrix buffer `max_buffer` and saves the result in
/// the first column of each row of the matrix buffer `result_buffer`. **This operation invalidates
/// the rest of the data in `result_buffer`**, but keeps constant `max_buffer`. That is:
/// If `max_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{11} & \cdots & a_{1n}\\
///                     \vdots &\ddots  & \vdots\\
///                     a_{m1} & \cdots & a_{mn}
///     \end{bmatrix}
/// ```
///
/// After calling `max_gpu_mat`, `result_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \max_{i}a_{1i} &  \cdots & \text{unexpected values} & \ldots\\
///                      \vdots & \cdots & \text{unexpected values} & \ldots\\
///                     \max_{i}a_{mi} & \cdots & \text{unexpected values} & \ldots\\
/// \end{bmatrix}
/// ```
///
/// `n_rows` is the number of rows in `max_buffer`. `n_cols` is the number of columns of
/// `max_buffer`. `max_work_size` is the maximum number of work items in a work group for the
/// selected device. `local_size` is the actual number of work items in each work group.
/// `num_groups` is the actual number of work groups and the number of columns in `result_buffer`.
fn max_gpu_mat(
    pro_que: &ProQue,
    max_buffer: &Buffer<f64>,
    result_buffer: &Buffer<f64>,
    n_rows: usize,
    mut n_cols: usize,
    max_work_size: usize,
    mut local_size: usize,
    mut num_groups: usize,
) {
    let kernel_max_gpu = pro_que
        .kernel_builder("max_gpu_mat_copy")
        .global_work_size((n_rows, n_cols))
        .local_work_size((1, local_size))
        .arg(max_buffer)
        .arg(result_buffer)
        .arg_local::<f64>(local_size)
        .arg(n_cols as u32)
        .build()
        .expect("Kernel max_gpu_mat_copy build failed.");

    unsafe {
        kernel_max_gpu
            .enq()
            .expect("Error while executing max_gpu_mat_copy kernel.");
    }

    n_cols = num_groups;
    local_size = if n_cols < max_work_size {
        n_cols
    } else {
        max_work_size
    };
    let matrix_actual_cols = num_groups;
    num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;

    while n_cols > 1 {
        let kernel_max_gpu = pro_que
            .kernel_builder("max_gpu_mat")
            .global_work_size((n_rows, n_cols))
            .local_work_size((1, local_size))
            .arg(result_buffer)
            .arg_local::<f64>(local_size)
            .arg(matrix_actual_cols as u32)
            .build()
            .expect("Kernel max_gpu_mat build failed.");

        unsafe {
            kernel_max_gpu
                .enq()
                .expect("Error while executing max_gpu_mat kernel.");
        }

        n_cols = num_groups;
        local_size = if n_cols < max_work_size {
            n_cols
        } else {
            max_work_size
        };
        num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;
    }
}

/// Sums all the elements of each row in the matrix buffer `sum_buffer` and saves the result in
/// the first column of each row (i.e. `max_buffer[i][0]`). **This operation invalidates
/// the rest of the data in `sum_buffer`**. That is:
/// If `sum_buffer` is equal to:
///
/// ```math
///     \begin{bmatrix} a_{11} & \cdots & a_{1n}\\
///                     \vdots &\ddots  & \vdots\\
///                     a_{m1} & \cdots & a_{mn}
///     \end{bmatrix}
/// ```
///
/// After calling `sum_gpu_mat`, `sum_buffer` will be equal to:
///
/// ```math
///     \begin{bmatrix} \sum_{i}^{n}a_{1i} &  \cdots & \text{unexpected values} & \ldots\\
///                      \vdots & \cdots & \text{unexpected values} & \ldots\\
///                     \sum_{i}^{n}a_{mi} & \cdots & \text{unexpected values} & \ldots\\
/// \end{bmatrix}
/// ```
///
/// `n_rows` is the number of rows in `max_buffer`. `n_cols` is the number of columns of
/// `max_buffer`. `max_work_size` is the maximum number of work items in a work group for the
/// selected device. `local_size` is the actual number of work items in each work group.
/// `num_groups` is the actual number of work groups and the number of columns in `result_buffer`.
fn sum_gpu_mat(
    pro_que: &ProQue,
    sum_buffer: &Buffer<f64>,
    n_rows: usize,
    mut n_cols: usize,
    max_work_size: usize,
    mut local_size: usize,
    mut num_groups: usize,
) {
    let n_cols_orig = n_cols as u32;
    while n_cols > 1 {
        let kernel_sum_gpu_mat = pro_que
            .kernel_builder("sum_gpu_mat")
            .global_work_size((n_rows, n_cols))
            .local_work_size((1, local_size))
            .arg(sum_buffer)
            .arg_local::<f64>(local_size)
            .arg(n_cols_orig)
            .build()
            .expect("Kernel sum_gpu_mat build failed.");

        unsafe {
            kernel_sum_gpu_mat
                .enq()
                .expect("Error while executing sum_gpu_mat kernel.");
        }

        n_cols = num_groups;
        local_size = if n_cols < max_work_size {
            n_cols
        } else {
            max_work_size
        };
        num_groups = (n_cols as f32 / local_size as f32).ceil() as usize;
    }
}
