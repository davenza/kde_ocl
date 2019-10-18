import pytest
import numpy as np
from scipy.stats import gaussian_kde
from kde_ocl import gaussian_kde_ocl


@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_pdf_1d(n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)

    train = np.random.normal(0, 1.8, n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.normal(0.5, 2.3, n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_scipy = gaussian_kde(train)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    pdf_scipy = kde_scipy(test[:100])
    pdf_ocl = kde_ocl(test)

    close = np.isclose(pdf_scipy, pdf_ocl[:100])

    assert close.all(), "Different pdf values: \n\tScipy values: " + str(pdf_scipy[~close]) + \
                        "\n\tocl values: " + str(pdf_ocl[:100][~close])

@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_pdf_2d(n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_scipy = gaussian_kde(train.T)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    pdf_scipy = kde_scipy(test[:100].T)
    pdf_ocl = kde_ocl(test)

    close = np.isclose(pdf_scipy, pdf_ocl[:100])

    assert close.all(), "Different pdf values: \n\tScipy values: " + str(pdf_scipy[~close]) + \
                        "\n\tocl values: " + str(pdf_ocl[:100][~close])

@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_logpdf_1d(n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.normal(0, 1.8, n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.normal(0.5, 2.3, n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_scipy = gaussian_kde(train)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    logpdf_scipy = kde_scipy.logpdf(test[:100])
    logpdf_ocl = kde_ocl.logpdf(test)

    close = np.isclose(logpdf_scipy, logpdf_ocl[:100])

    assert close.all(), "Different pdf values: \n\tScipy values: " + str(logpdf_scipy[~close]) + \
                        "\n\tocl values: " + str(logpdf_ocl[:100][~close])

@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_logpdf_2d(n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_scipy = gaussian_kde(train.T)
    kde_ocl = gaussian_kde_ocl(train)

    # Check only the first 100 instances because Scipy can be too slow.
    logpdf_scipy = kde_scipy.logpdf(test[:100].T)
    logpdf_ocl = kde_ocl.logpdf(test)

    close = np.isclose(logpdf_scipy, logpdf_ocl[:100])

    assert close.all(), "Different logpdf values: \n\tScipy values: " + str(logpdf_scipy[~close]) + \
                        "\n\tocl values: " + str(logpdf_ocl[:100][~close])

@pytest.mark.times_ocl
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_pdf_2d_benchmark_ocl(benchmark, n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_ocl = gaussian_kde_ocl(train)

    benchmark(kde_ocl.pdf, test)

@pytest.mark.times_scipy
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_pdf_2d_benchmark_scipy(benchmark, n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_ocl = gaussian_kde(train.T)

    benchmark(kde_ocl.pdf, test.T)

@pytest.mark.times_ocl
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (1000, 10000000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_logpdf_2d_benchmark_ocl(benchmark, n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_ocl = gaussian_kde_ocl(train)

    benchmark(kde_ocl.logpdf, test)

@pytest.mark.times_scipy
@pytest.mark.times
@pytest.mark.parametrize("n_train, n_test", [(100000, 1000), (100, 10000)])
@pytest.mark.parametrize("rowmajor_train", [True, False])
@pytest.mark.parametrize("rowmajor_test", [True, False])
def test_logpdf_2d_benchmark_scipy(benchmark, n_train, n_test, rowmajor_train, rowmajor_test):
    np.random.seed(0)
    train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
    if not rowmajor_train:
        train = np.asfortranarray(train)
    test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)
    if not rowmajor_test:
        test = np.asfortranarray(test)

    kde_ocl = gaussian_kde(train.T)

    benchmark(kde_ocl.logpdf, test.T)