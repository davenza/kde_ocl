/**
##########################################
################  MISC  ##################
##########################################
*/


__kernel void fill_value(__global double *vec, __private double value) {
    vec[get_global_id(0)] = value;
}

__kernel void fill_value_uint(__global uint *vec, __private uint value) {
    vec[get_global_id(0)] = value;
}

__kernel void sum_vectors(__global double *left, __constant double *right) {
    uint idx = get_global_id(0);
    left[idx] += right[idx];
}


/**
##########################################
###############  COMMON  #################
##########################################
 */

__kernel void substract(__constant double *train_data,
                        __constant double *vec,
                        __global double *res,
                        __private uint row,
                        __private uint n_col) {
    int i = get_global_id(0);
    int c = i % n_col;
    res[i] = train_data[i] - vec[row*n_col + c];
}

__kernel void solve(__global double *diff_data, __constant double *chol, __private uint n_col) {
    uint r = get_global_id(0);
    uint index_row = r * n_col;

    for (uint c = 0; c < n_col; c++) {
        for (uint i = 0; i < c; i++) {
            diff_data[index_row + c] -= chol[c * n_col + i] * diff_data[index_row + i];
        }
        diff_data[index_row + c] /= chol[c * n_col + c];
    }
}

__kernel void square(__global double *solve_data) {
    uint idx = get_global_id(0);
    solve_data[idx] = solve_data[idx] * solve_data[idx];
}

/**
##########################################
#################  PDF  ##################
##########################################
*/

__kernel void sumout(__constant double *square_data,
                    __global double *sol_vec,
                    __private uint n_col,
                    __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = exp(-0.5 * sol_vec[r] - lognorm_factor);
}

__kernel void sum_gpu_vec(__global double *input,
                          __local double *localSums) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localSums[local_id] = input[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localSums[local_id] += localSums[local_id + stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localSums[local_id+1] += localSums[local_id+1 + stride];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        input[group_id] = localSums[0];
    }
}

/**
##########################################
########  logPDF - Iterate test  #########
##########################################
*/

__kernel void logsumout(__constant double *square_data,
                        __global double *sol_vec,
                        __private uint n_col,
                        __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = (-0.5 * sol_vec[r]) - lognorm_factor;
}

__kernel void max_gpu_vec_copy(__constant double *input,
                               __global double *maxGroups,
                               __local double *localMaxs) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localMaxs[local_id] = input[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[group_id] = localMaxs[0];
    }
}

__kernel void max_gpu_vec(__global double* maxGroups,
                     __local double *localMaxs) {

    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[group_id] = localMaxs[0];
    }
}

__kernel void log_sum_gpu_vec(__global double *input,
                          __local double *localSums,
                          __constant double *maxexp) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    if (group_id == num_groups) {
        group_size = get_global_size(0) - group_id*group_size;
    }

    localSums[local_id] = exp(input[global_id]-maxexp[0]);

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localSums[local_id] += localSums[local_id + stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localSums[local_id+1] += localSums[local_id+1 + stride];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        input[group_id] = localSums[0];
    }
}

__kernel void copy_logpdf_result(__constant double *logsum,
                                 __constant double *maxexp,
                                 __global double *res,
                                 __private uint res_offset) {
    res[res_offset] = maxexp[0] + log(logsum[0]);
}


/**
##########################################
## logPDF - Iterate train (low memory) ###
##########################################
*/

__kernel void logsumout_checkmax(__constant double *square_data,
                                __global double *sol_vec,
                                __global double *max_vec,
                                __private uint n_col,
                                __private double lognorm_factor) {
    uint r = get_global_id(0);
    uint idx = r * n_col;

    sol_vec[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_vec[r] += square_data[idx + i];
    }

    sol_vec[r] = (-0.5 * sol_vec[r]) - lognorm_factor;
    max_vec[r] = max(max_vec[r], sol_vec[r]);
}

__kernel void exp_and_sum(__constant double* logsum, __constant double* maxexp, __global double *res) {
    uint idx = get_global_id(0);
    res[idx] += exp(logsum[idx] - maxexp[idx]);
}

__kernel void log_and_sum(__global double* res, __constant double* maxexp) {
    uint idx = get_global_id(0);
    res[idx] = log(res[idx]) + maxexp[idx];
}



/**
##########################################
## logPDF - Iterate train (high memory) ##
##########################################
*/

__kernel void logsumout_to_matrix(__constant double *square_data,
                                    __global double *sol_mat,
                                    __private uint n_col,
                                    __private uint sol_row,
                                    __private uint n_train_instances,
                                    __private double lognorm_factor) {
    uint r = n_train_instances*get_global_id(0) + sol_row;
    uint idx = get_global_id(0) * n_col;

    sol_mat[r] = square_data[idx];
    for (uint i = 1; i < n_col; i++) {
        sol_mat[r] += square_data[idx + i];
    }

    sol_mat[r] = (-0.5 * sol_mat[r]) - lognorm_factor;
}

__kernel void max_gpu_mat_copy(__constant double *input,
                               __global double* maxGroups,
                               __local double *localMaxs,
                               __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = input[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id+stride]);
            }
            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1], localMaxs[local_id+1+stride]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*num_groups+group_id] = localMaxs[0];
    }
}

__kernel void max_gpu_mat(__global double* maxGroups,
                          __local double *localMaxs,
                          __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] = max(localMaxs[local_id], localMaxs[local_id + stride]);
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] = max(localMaxs[local_id+1 + stride], localMaxs[local_id+1]);
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*array_n_cols + group_id] = localMaxs[0];
    }
}


__kernel void exp_and_sum_mat(__global double* res, __constant double* maxexp, __private uint num_groups) {
    uint row = get_global_id(0);
    uint col = get_global_id(1);
    uint n_col = get_global_size(1);
    res[row*n_col+col] = exp(res[row*n_col+col] - maxexp[row*num_groups]);
}

__kernel void sum_gpu_mat(__global double* maxGroups,
                          __local double *localMaxs,
                          __private uint array_n_cols) {

    uint global_id_row = get_global_id(0);
    uint global_id_col = get_global_id(1);
    uint n_cols = get_global_size(1);
    uint local_id = get_local_id(1);
    uint group_size = get_local_size(1);
    uint group_id = get_group_id(1);
//   FIXME: This code returns num_groups = 3 for global_size = 1000 and local_size = 256, so it does not work as expected
//      when local_work_size does not evenly divide global_work_size.
//    uint num_groups = get_num_groups(1);

    //This is equal to ceil(n_cols/group_size): https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    uint num_groups = (n_cols + group_size - 1) / group_size;

    if (group_id+1 == num_groups) {
        group_size = get_global_size(1) - group_id*group_size;
    }

    localMaxs[local_id] = maxGroups[global_id_row*array_n_cols + global_id_col];

    while (group_size > 1) {
        int stride = group_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (group_size % 2 == 0) {
            if (local_id < stride) {
                localMaxs[local_id] += localMaxs[local_id+stride];
            }

            group_size = group_size / 2;
        }
        else {
            if (local_id < stride) {
                localMaxs[local_id+1] += localMaxs[local_id+stride+1];
            }
            group_size = (group_size / 2) + 1;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (local_id == 0) {
        maxGroups[global_id_row*array_n_cols + group_id] = localMaxs[0];
    }
}

__kernel void log_and_sum_mat(__global double* res,
                                __constant double *summed_mat,
                                __constant double* maxexp,
                                __private uint n_col,
                                __private uint num_groups) {
    uint idx = get_global_id(0);
    res[idx] = log(summed_mat[idx*n_col]) + maxexp[idx*num_groups];
}

