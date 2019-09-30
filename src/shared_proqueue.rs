use ocl::{Buffer, ProQue};
use libc::c_double;
use std::fs::File;
use std::io::Read;
use std::slice;
use std::ptr;
use std::f64;

use {DoubleNumpyArray, Error, load_numpy_2d, lognorm_factor, get_max_work_size, sum_gpu_vec,
     buffer_fill_value, max_gpu_vec_copy, log_sum_gpu_vec, max_gpu_mat, sum_gpu_mat};

/// Implements a KDE density model in OpenCL.
#[repr(C)]
#[derive(Debug)]
pub struct SharedGaussianKDE {
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

#[no_mangle]
pub unsafe extern "C" fn new_proque() -> *mut ProQue {
    // TODO: The OpenCL code should be included in the code to make easier distribute the library.
    let mut f = File::open("src/kde_gaussian.cl")
        .expect("OpenCL code file not found");
    let mut src = String::new();
    f.read_to_string(&mut src).expect("Error while reading OpenCL code file.");

    let pro_que = ProQue::builder()
        .src(src)
        .build().expect("Error while creating OpenCL ProQue.");

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
pub unsafe extern "C" fn gaussian_kde_shared_init(pro_que: *mut ProQue,
                                                  chol: *const DoubleNumpyArray,
                                                  training_data: *const DoubleNumpyArray,
                                                  error: *mut Error) -> *mut SharedGaussianKDE {
    let n = *(*training_data).shape;
    let d = *(*chol).shape;
    let chol_cov = load_numpy_2d(chol);
    let lognorm_factor = lognorm_factor(n, d, &chol_cov);
    let chol_vec = chol_cov.into_raw_vec();

    let pro_que = Box::from_raw(pro_que);

    let training_slice = slice::from_raw_parts((*training_data).ptr, (*training_data).size);

    let (training_buffer, chol_buffer) =
        copy_buffers!(pro_que, error, training_slice, chol_vec => ptr::null_mut());

    let kde = Box::new(SharedGaussianKDE {
        n,
        d,
        training_data: training_buffer,
        chol_cov: chol_buffer,
        lognorm_factor
    });

    let ptr_kde = Box::into_raw(kde);

    Box::into_raw(pro_que);

    *error = Error::NoError;
    ptr_kde
}

/// Frees the `KDEDensityOcl`.
#[no_mangle]
pub extern "C" fn gaussian_shared_free(kde: *mut SharedGaussianKDE, pro_que: *mut ProQue) {
    gaussian_kde_shared_free(kde);
    gaussian_proque_free(pro_que);
}

/// Frees the `KDEDensityOcl`.
#[no_mangle]
pub extern "C" fn gaussian_kde_shared_free(kde: *mut SharedGaussianKDE) {
    if kde.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(kde);
    }
}

/// Frees the `KDEDensityOcl`.
#[no_mangle]
pub extern "C" fn gaussian_proque_free(pro_que: *mut ProQue) {
    if pro_que.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(pro_que);
    }
}

#[no_mangle]
pub unsafe extern "C" fn gaussian_kde_shared_pdf(kde: *mut SharedGaussianKDE,
                                          pro_que: *mut ProQue,
                                          testing_data: *const DoubleNumpyArray,
                                          result: *mut c_double,
                                          error: *mut Error) {

    let kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);

    let m = *(*testing_data).shape;
    let final_result = slice::from_raw_parts_mut(result, m);

    let max_work_size = get_max_work_size(&pro_que);

    let d = kde_box.d;
//    Iterates over training or testing data?
    let len_iteration = if kde_box.n >= m { kde_box.n } else { m };

    pro_que.set_dims(len_iteration*d);

    let local_size = if len_iteration < max_work_size { len_iteration } else { max_work_size };
    let num_groups = (len_iteration as f32 / local_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*testing_data).ptr, m*d);

    let (test_instances_buffer,) =
        copy_buffers!(pro_que, error, test_slice);

    let (final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        empty_buffers!(pro_que, error, f64, m, len_iteration*d, len_iteration);

    let kernel_substract = {
        let (matrix_buf, vec_buf) =
            if kde_box.n >= m {
                (&kde_box.training_data, &test_instances_buffer)
            } else {
                (&test_instances_buffer, &kde_box.training_data)
            };

        pro_que.kernel_builder("substract")
            .arg(matrix_buf)
            .arg(vec_buf)
            .arg(&tmp_matrix_buffer)
            .arg_named("row", &0u32)
            .arg(d as u32)
            .build().expect("Kernel substract build failed.")
    };

    let kernel_solve = pro_que.kernel_builder("solve")
        .global_work_size(len_iteration)
        .arg(&tmp_matrix_buffer)
        .arg(&kde_box.chol_cov)
        .arg(d as u32)
        .build().expect("Kernel solve build failed.");

    let kernel_square = pro_que.kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build().expect("Kernel square build failed.");

    let kernel_sumout = pro_que.kernel_builder("sumout")
        .global_work_size(len_iteration)
        .arg(&tmp_matrix_buffer)
        .arg(&tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde_box.lognorm_factor)
        .build().expect("Kernel sumout build failed.");

    if kde_box.n >= m {
        for i in 0..m {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract.enq().expect("Error while executing substract kernel.");
            kernel_solve.enq().expect("Error while executing solve kernel.");
            kernel_square.enq().expect("Error while executing square kernel.");
            kernel_sumout.enq().expect("Error while executing sumout kernel.");
            sum_gpu_vec(&pro_que, &tmp_vec_buffer,
                        len_iteration, max_work_size, local_size, num_groups);

            tmp_vec_buffer.copy(&final_result_buffer, Some(i), Some(1))
                .queue(pro_que.queue())
                .enq().expect("Error copying to result buffer.");
        }
    } else {
        buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

        for i in 0..kde_box.n {
            kernel_substract.set_arg("row", i as u32).unwrap();
            kernel_substract.enq().expect("Error while executing substract kernel.");
            kernel_solve.enq().expect("Error while executing solve kernel.");
            kernel_square.enq().expect("Error while executing square kernel.");
            kernel_sumout.enq().expect("Error while executing sumout kernel.");

            let kernel_sumvectors = pro_que.kernel_builder("sum_vectors")
                .global_work_size(m)
                .arg(&final_result_buffer)
                .arg(&tmp_vec_buffer)
                .build().expect("Kernel sum_vectors build failed.");

            kernel_sumvectors.enq().expect("Error while executing sum_vectors kernel.");
        }

    }

    final_result_buffer.cmd().queue(pro_que.queue())
        .read(final_result).enq().expect("Error reading result data.");
    *error = Error::NoError;
    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}


#[no_mangle]
pub unsafe extern "C"  fn gaussian_kde_shared_logpdf(kde: *mut SharedGaussianKDE,
                                              pro_que: *mut ProQue,
                                              x: *const DoubleNumpyArray,
                                              result: *mut c_double,
                                              error: *mut Error) {
    let mut kde_box = Box::from_raw(kde);
    let mut pro_que = Box::from_raw(pro_que);
    let m = *(*x).shape;
    if kde_box.n >= m {
        logpdf_iterate_shared_test(&mut kde_box, &mut pro_que, x, result, error);
    } else {
        logpdf_iterate_shared_train(&mut kde_box, &mut pro_que, x, result, error);
    }

    *error = Error::NoError;
    Box::into_raw(kde_box);
    Box::into_raw(pro_que);
}

/// We iterate over the test points if there are more training points.
unsafe fn logpdf_iterate_shared_test(kde: &mut Box<SharedGaussianKDE>,
                              pro_que: &mut Box<ProQue>,
                              x: *const DoubleNumpyArray,
                              result: *mut c_double,
                              error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;
    let final_result = slice::from_raw_parts_mut(result, m);
    let max_work_size = get_max_work_size(&pro_que);

    let n = kde.n;
    pro_que.set_dims(n*d);

    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m*d);

    let (test_instances_buffer,) =
        copy_buffers!(pro_que, error, test_slice);

    let (max_buffer, final_result_buffer, tmp_matrix_buffer, tmp_vec_buffer) =
        empty_buffers!(pro_que, error, f64, num_groups, m, n*d, n);

    let kernel_substract = pro_que.kernel_builder("substract")
        .arg(&kde.training_data)
        .arg(&test_instances_buffer)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build().expect("Kernel substract build failed.");

    let kernel_solve = pro_que.kernel_builder("solve")
        .global_work_size(n)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build().expect("Kernel solve build failed.");

    let kernel_square = pro_que.kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build().expect("Kernel square build failed.");

    let kernel_sumout = pro_que.kernel_builder("logsumout")
        .global_work_size(n)
        .arg(&tmp_matrix_buffer)
        .arg(&tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build().expect("Kernel logsumout build failed.");

    for i in 0..m {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout.enq().expect("Error while executing logsumout kernel.");

        max_gpu_vec_copy(&pro_que, &tmp_vec_buffer, &max_buffer,
                         n, max_work_size, local_work_size, num_groups);


        log_sum_gpu_vec(&pro_que, &tmp_vec_buffer, &max_buffer,
                        n, max_work_size, local_work_size, num_groups);

        let kernel_log_sum_gpu = pro_que.kernel_builder("copy_logpdf_result")
            .global_work_size(1)
            .arg(&tmp_vec_buffer)
            .arg(&max_buffer)
            .arg(&final_result_buffer)
            .arg(i as u32)
            .build().expect("Kernel copy_logpdf_result build failed.");

        kernel_log_sum_gpu.enq()
            .expect("Error while executing copy_logpdf_result kernel.");

    }
    final_result_buffer.cmd().queue(pro_que.queue())
        .read(final_result).enq().expect("Error reading result data.");
}


unsafe fn logpdf_iterate_shared_train(kde: &mut Box<SharedGaussianKDE>,
                                     pro_que: &mut Box<ProQue>,
                                     x: *const DoubleNumpyArray,
                                     result: *mut c_double,
                                     error: *mut Error) {
    let m = *(*x).shape;
    let tmp_vec_buffer = Buffer::<f64>::builder()
        .context(pro_que.context())
        .len(m*kde.n)
        .build();

    match tmp_vec_buffer {
        Ok(b) => logpdf_iterate_shared_train_high_memory(kde, pro_que, x, result, &b, error),
        Err(_) => {
            let (tmp_vec_buffer,) = empty_buffers!(pro_que, error, f64, m);
            // TODO: If n < 2m, is it better to iterate over the training data?
            logpdf_iterate_shared_train_low_memory(kde, pro_que, x, result, &tmp_vec_buffer, error);
        }
    }
}

unsafe fn logpdf_iterate_shared_train_low_memory(kde: &mut Box<SharedGaussianKDE>,
                                          pro_que: &mut Box<ProQue>,
                                          x: *const DoubleNumpyArray,
                                          result: *mut c_double,
                                          tmp_vec_buffer: &Buffer<f64>,
                                          error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;

    let final_result = slice::from_raw_parts_mut(result, m);

    let n = kde.n;
    pro_que.set_dims(m*d);

    let test_slice = slice::from_raw_parts((*x).ptr, m*d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);

    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        empty_buffers!(pro_que, error, f64, m, m, m*d);

    buffer_fill_value(&pro_que, &max_buffer, m, f64::MIN);
    buffer_fill_value(&pro_que, &final_result_buffer, m, 0.0f64);

    let kernel_substract = pro_que.kernel_builder("substract")
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build().expect("Kernel substract build failed.");

    let kernel_solve = pro_que.kernel_builder("solve")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build().expect("Kernel solve build failed.");

    let kernel_square = pro_que.kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build().expect("Kernel square build failed.");

    let kernel_sumout = pro_que.kernel_builder("logsumout")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build().expect("Kernel logsumout build failed.");

    let kernel_sumout_checkmax = pro_que.kernel_builder("logsumout_checkmax")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(d as u32)
        .arg(kde.lognorm_factor)
        .build().expect("Kernel logsumout_checkmax build failed.");

    let kernel_expsum = pro_que.kernel_builder("exp_and_sum")
        .global_work_size(m)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(&final_result_buffer)
        .build().expect("Kernel exp_and_sum build failed.");

    let kernel_log_and_sum = pro_que.kernel_builder("log_and_sum")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(&max_buffer)
        .build().expect("Kernel log_and_sum build failed.");

    // Writes the max loglikelihoods in the max_buffer
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout_checkmax.enq().expect("Error while executing logsumout_checkmax kernel.");
    }

    // Computes the loglikelihood using the max_buffer.
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout.enq().expect("Error while executing logsumout kernel.");
        kernel_expsum.enq().expect("Error while executing exp_and_sum kernel.");
    }
    kernel_log_and_sum.enq().expect("Error while executing log_and_sum kernel.");
    final_result_buffer.cmd().queue(pro_que.queue())
        .read(final_result).enq().expect("Error reading result data.");

}

unsafe fn logpdf_iterate_shared_train_high_memory(kde: &mut Box<SharedGaussianKDE>,
                                                 pro_que: &mut Box<ProQue>,
                                                 x: *const DoubleNumpyArray,
                                                 result: *mut c_double,
                                                 tmp_vec_buffer: &Buffer<f64>,
                                                 error: *mut Error) {
    let m = *(*x).shape;
    let d = kde.d;

    let final_result = slice::from_raw_parts_mut(result, m);

    let n = kde.n;
    pro_que.set_dims(m*d);

    let max_work_size = get_max_work_size(&pro_que);
    let local_work_size = if n < max_work_size { n } else { max_work_size };
    let num_groups = (n as f32 / local_work_size as f32).ceil() as usize;

    let test_slice = slice::from_raw_parts((*x).ptr, m*d);

    let (test_instances_buffer,) = copy_buffers!(pro_que, error, test_slice);
    let (max_buffer, final_result_buffer, tmp_matrix_buffer) =
        empty_buffers!(pro_que, error, f64, m*num_groups, m, m*d);

    let kernel_substract = pro_que.kernel_builder("substract")
        .arg(&test_instances_buffer)
        .arg(&kde.training_data)
        .arg(&tmp_matrix_buffer)
        .arg_named("row", &0u32)
        .arg(d as u32)
        .build().expect("Kernel substract build failed.");

    let kernel_solve = pro_que.kernel_builder("solve")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(&kde.chol_cov)
        .arg(d as u32)
        .build().expect("Kernel solve build failed.");

    let kernel_square = pro_que.kernel_builder("square")
        .arg(&tmp_matrix_buffer)
        .build().expect("Kernel square build failed.");

    let kernel_sumout = pro_que.kernel_builder("logsumout_to_matrix")
        .global_work_size(m)
        .arg(&tmp_matrix_buffer)
        .arg(tmp_vec_buffer)
        .arg(d as u32)
        .arg_named("sol_row", &0u32)
        .arg(n as u32)
        .arg(kde.lognorm_factor)
        .build().expect("Kernel logsumout_to_matrix build failed.");

    let kernel_exp_and_sum = pro_que.kernel_builder("exp_and_sum_mat")
        .global_work_size((m, n))
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(num_groups as u32)
        .build().expect("Kernel exp_and_sum_mat build failed.");

    let kernel_log_and_sum = pro_que.kernel_builder("log_and_sum_mat")
        .global_work_size(m)
        .arg(&final_result_buffer)
        .arg(tmp_vec_buffer)
        .arg(&max_buffer)
        .arg(n as u32)
        .arg(num_groups as u32)
        .build().expect("Kernel log_and_sum_mat build failed.");

    // Writes the max loglikelihoods in the max_buffer
    for i in 0..n {
        kernel_substract.set_arg("row", i as u32).unwrap();
        kernel_substract.enq().expect("Error while executing substract kernel.");
        kernel_solve.enq().expect("Error while executing solve kernel.");
        kernel_square.enq().expect("Error while executing square kernel.");
        kernel_sumout.set_arg("sol_row", i as u32).unwrap();
        kernel_sumout.enq().expect("Error while executing logsumout_to_matrix kernel.");
    }

    max_gpu_mat(&pro_que, tmp_vec_buffer, &max_buffer,
                m, n, max_work_size, local_work_size, num_groups);
    kernel_exp_and_sum.enq().expect("Error while executing exp_and_sum_mat kernel.");
    sum_gpu_mat(&pro_que, tmp_vec_buffer, m, n,
                max_work_size, local_work_size, num_groups);

    kernel_log_and_sum.enq().expect("Error while executing kernel log_and_sum_mat kernel.");
    final_result_buffer.cmd().queue(pro_que.queue())
        .read(final_result).enq().expect("Error reading result data.");

}