extern crate cbindgen;

use std::env;

use std::fs;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut config: cbindgen::Config = cbindgen::Config::from_file("cbindgen.toml").unwrap();
    config.language = cbindgen::Language::C;
    cbindgen::generate_with_config(&crate_dir, config)
        .unwrap()
        .write_to_file("target/kde-ocl-sys.h");



    let src = fs::read_to_string("src/kde_gaussian.cl").expect("Error while reading OpenCL code file.");
    let rust_cl_code = format!("pub const OPEN_CL_CODE: &str = \"{}\";", src);
    fs::write("src/open_cl_code.rs", rust_cl_code).expect("Unable to write OpenCL Rust code");
}