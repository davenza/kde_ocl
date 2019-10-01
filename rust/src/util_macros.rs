/// This macro copies some slice-like data into OpenCL buffers. The buffers are returned in a tuple
/// containing all the buffers. If some buffer returns with error while creating it, it returns
/// setting a memory error. There are two variants of the macro, depending on the return value if
/// the allocation fails:
/// * The simpler variant just makes a `return;`
/// * The more complex variant adds a `=> $ret` suffix to return with a `return $ret;`
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate kde_rust;
/// // Simple variant
/// fn main() {
///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
///     let mut error = Error::NoError;
///     let error_ptr = &mut error as *mut Error;
///     let vec = vec![0.0f64; 10];
///     let vec2 = vec![1.0f64; 20];
///     // It does a return; if the allocation fails:
///     let (buffer,) = copy_buffers!(pro_que, error_ptr, vec);
///     // Multiple buffering copies:
///     let (buffer1, buffer2) = copy_buffers!(pro_que, error_ptr, vec, vec2);
/// }
/// ```
/// ```
/// # #[macro_use] extern crate kde_rust;
///  // Complex variant.
///  fn dummy() -> bool {
///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
///     let mut error = Error::NoError;
///     let error_ptr = &mut error as *mut Error;
///     let vec = vec![0.0f64; 10];
///     let vec2 = vec![1.0f64; 20];
///     // It does a  return true; if the allocation fails:
///     let (buffer,) = copy_buffers!(pro_que, error_ptr, vec => true);
///     // Multiple buffering copies:
///     let (buffer1, buffer2) = copy_buffers!(pro_que, error_ptr, vec, vec2 => true);
///  }
/// ```
///
#[macro_export]
macro_rules! copy_buffers {
    ($pro_que:expr, $error:expr, $($slice:expr),+) => {
        {
            (
                $(
                    match Buffer::builder()
                        .context($pro_que.context())
                        .len($slice.len())
                        .copy_host_slice(&$slice)
                        .build() {
                            Ok(b) => b,
                            Err(_) => {
                                *$error = Error::MemoryError;
                                return;
                            }
                        },
                )+

            )
        }
    };

    ($pro_que:expr, $error:expr, $($slice:expr),+ => $ret:expr) => {
        {
            (
                $(
                    match Buffer::builder()
                        .context($pro_que.context())
                        .len($slice.len())
                        .copy_host_slice(&$slice)
                        .build() {
                            Ok(b) => b,
                            Err(_) => {
                                *$error = Error::MemoryError;
                                return $ret;
                            }
                        },
                )+

            )
        }
    };
}

/// This macro creates some new OpenCL buffer, given its type and their lengths. The buffers are
/// returned in a tuple containing all the buffers. If some buffer returns with error while
/// creating it, it returns setting a memory error.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate kde_rust;
/// fn main() {
///     let pro_que = ProQue::builder().src(r#"__kernel void dummy {}"#).build().unwrap();
///     let mut error = Error::NoError;
///     let error_ptr = &mut error as *mut Error;
///     // It does a return; if the allocation fails.
///     let (buffer,) = empty_buffers!(pro_que, error_ptr, f64, 10);
///     // Multiple buffer allocations
///     let (buffer, buffer2, buffer3) = empty_buffers!(pro_que, error_ptr, f64, 10, 35, 40);
/// }
/// ```
#[macro_export]
macro_rules! empty_buffers {
    ($pro_que:expr, $error:expr, $type:ty, $($len:expr),+) => {
        {

            (
                $(
                    match Buffer::<$type>::builder().context($pro_que.context()).len($len).build() {
                        Ok(b) => b,
                        Err(_) => {
                            *$error = Error::MemoryError;
                            return;
                        }
                    },
                )+
            )
        }
    };
}
