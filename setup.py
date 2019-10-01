import distutils.cmd
import setuptools
import os
import shutil

long_description = """This repository implements Gaussian [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) using 
OpenCL to achieve important performance gains:


The Python interface is based on the [Scipy's `gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) class, 
so it should be pretty easy to replace the CPU implementation of `gaussian_kde` with the
OpenCL implementation in this repository `gaussian_kde_ocl`."""

def build_native(spec):
    # build an example rust library
    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
        path='./rust'
    )

    spec.add_cffi_module(
        module_path='kde_ocl._ffi',
        dylib=lambda: build.find_dylib('kde_ocl_sys', in_path='target/release'),
        header_filename=lambda: build.find_header('kde-ocl-sys.h', in_path='target'),
        rtld_flags=['NOW', 'NODELETE']
    )

class CleanCommand(distutils.cmd.Command):
    """
    Our custom command to clean out junk files.
    """
    description = "Cleans out all generated files and folders."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        files_to_delete = ["kde_ocl/_ffi.py", "kde_ocl/_ffi__ffi.py", "kde_ocl/_ffi__lib.so", "rust/src/open_cl_code.rs"]

        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

        current_dir = os.getcwd()

        os.chdir('rust')
        os.system("cargo clean")
        os.chdir(current_dir)

        folders_to_delete = [".eggs", "kde_ocl.egg-info", "build"]

        for folder in folders_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)

setuptools.setup(
    name='kde_ocl',
    version='0.1',
    author="David Atienza",
    author_email="datienza@fi.upm.es",
    description="An OpenCL implementation of Kernel Density Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javatechy/dokr",
    packages=['kde_ocl'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    platforms='any',
    setup_requires=['milksnake'],
    install_requires=['milksnake', "numpy", "six"],
    milksnake_tasks=[
        build_native
    ],
    cmdclass={
        'clean': CleanCommand,
    },
)