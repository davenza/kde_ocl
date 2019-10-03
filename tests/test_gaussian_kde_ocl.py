import pytest
import numpy as np
from scipy.stats import gaussian_kde
from kde_ocl import gaussian_kde_ocl


@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
def test_pdf_1d(n_train, n_test):
    np.random.seed(0)
    train = np.random.normal(0, 1.8, n_train)
    test = np.random.normal(0.5, 2.3, n_test)

    kde_scipy = gaussian_kde(train)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    pdf_scipy = kde_scipy(test[:100])
    pdf_ocl = kde_ocl(test)

    assert np.isclose(pdf_scipy, pdf_ocl[:100]).all()

@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
def test_pdf_2d(n_train, n_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

    kde_scipy = gaussian_kde(train.T)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    pdf_scipy = kde_scipy(test[:100].T)
    pdf_ocl = kde_ocl(test)

    assert np.isclose(pdf_scipy, pdf_ocl[:100]).all()


@pytest.mark.times_ocl
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
def test_pdf_2d_benchmark_ocl(benchmark, n_train, n_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

    kde_ocl = gaussian_kde_ocl(train)

    benchmark(kde_ocl.pdf, test)

@pytest.mark.times_scipy
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
def test_pdf_2d_benchmark_scipy(benchmark, n_train, n_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

    kde_ocl = gaussian_kde(train.T)

    benchmark(kde_ocl.pdf, test.T)

@pytest.mark.times_ocl
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
def test_logpdf_2d_benchmark_ocl(benchmark, n_train, n_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

    kde_ocl = gaussian_kde_ocl(train)

    benchmark(kde_ocl.logpdf, test)

@pytest.mark.times_scipy
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (100, 10000)])
def test_logpdf_2d_benchmark_scipy(benchmark, n_train, n_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

    kde_ocl = gaussian_kde(train.T)

    benchmark(kde_ocl.logpdf, test.T)
