from __future__ import print_function
import gaussian_kde_ocl
import numpy as np
import time
import scipy.stats as spstats


def compareResults(truth, ocl):
    if np.isclose(truth, ocl).all():
        print("All values close")
    else:
        print("Some values are different (mean error):", np.abs(truth - ocl).mean())
        print("Count of different values: ", (~np.isclose(truth, ocl)).sum())


print("====================================")
print("=========== Train > test ===========")
print("====================================")
n_train = 100000
n_test = 1000
print(n_train, "train instances.", n_test, "test instances.")
train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

pdf_scipy = spstats.gaussian_kde(train.T)

kde_normal = gaussian_kde_ocl.gaussian_kde_ocl(train)
start = time.time()
pdf_ocl = kde_normal(test)
end = time.time()
print("OpenCL time:", (end-start))

shared_kde_normal = gaussian_kde_ocl.shared_gaussian_kde_ocl(train)
start = time.time()
shared_pdf_ocl = shared_kde_normal(test)
end = time.time()
print("OpenCL shared time:", (end-start))

start = time.time()
logpdf_ocl = kde_normal.logpdf(test)
end = time.time()
print("OpenCL logtime:", (end-start))

start = time.time()
shared_logpdf_ocl = shared_kde_normal.logpdf(test)
end = time.time()
print("OpenCL shared logtime:", (end-start))

start = time.time()
pdf_truth = pdf_scipy.pdf(test.T)
end = time.time()
print("Python/Scipy time:", (end-start))

start = time.time()
logpdf_truth = pdf_scipy.logpdf(test.T)
end = time.time()
print("Python/Scipy log time:", (end-start))
print()

print("Comparing OpenCL with Python/Scipy results...")
print("pdf:")
compareResults(pdf_truth, pdf_ocl)
print("logpdf:")
compareResults(logpdf_truth, logpdf_ocl)

print("Shared pdf:")
compareResults(pdf_truth, shared_pdf_ocl)
print("Shared logpdf:")
compareResults(logpdf_truth, shared_logpdf_ocl)

print("====================================")
print("=== Test > train (large dataset) ===")
print("====================================")
n_train = 1000
n_test = 10000000
print(n_train, "train instances.", n_test, "test instances.")
train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

pdf_scipy = spstats.gaussian_kde(train.T)

kde_normal = gaussian_kde_ocl.gaussian_kde_ocl(train)
start = time.time()
pdf_ocl = kde_normal(test)
end = time.time()
print("OpenCL time:", (end-start))

shared_kde_normal = gaussian_kde_ocl.shared_gaussian_kde_ocl(train)
start = time.time()
shared_pdf_ocl = kde_normal(test)
end = time.time()
print("OpenCL shared time:", (end-start))

start = time.time()
logpdf_ocl = kde_normal.logpdf(test)
end = time.time()
print("OpenCL logtime:", (end-start))

start = time.time()
shared_logpdf_ocl = shared_kde_normal.logpdf(test)
end = time.time()
print("OpenCL shared logtime:", (end-start))

start = time.time()
# Don't test with every point or the Python code will throw OutOfMemory.
pdf_truth = pdf_scipy.pdf(test[:100].T)
end = time.time()
print("Python/Scipy time:", (end-start))

start = time.time()
logpdf_truth = pdf_scipy.logpdf(test[:100].T)
end = time.time()
print("Python/Scipy log time:", (end-start))
print()

print("Comparing OpenCL with Python/Scipy results...")
print("pdf:")
compareResults(pdf_truth, pdf_ocl[:100])
print("logpdf:")
compareResults(logpdf_truth, logpdf_ocl[:100])

print("Shared pdf:")
compareResults(pdf_truth, shared_pdf_ocl[:100])
print("Shared logpdf:")
compareResults(logpdf_truth, shared_logpdf_ocl[:100])


print("====================================")
print("=== Test > train (small dataset) ===")
print("====================================")
n_train = 100
n_test = 10000
print(n_train, "train instances.", n_test, "test instances.")
train = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_train)
test = np.random.multivariate_normal([0,0], [[3,2],[2,3]], n_test)

pdf_scipy = spstats.gaussian_kde(train.T)

kde_normal = gaussian_kde_ocl.gaussian_kde_ocl(train)
start = time.time()
pdf_ocl = kde_normal(test)
end = time.time()
print("OpenCL time:", (end-start))

shared_kde_normal = gaussian_kde_ocl.shared_gaussian_kde_ocl(train)
start = time.time()
shared_pdf_ocl = shared_kde_normal(test)
end = time.time()
print("OpenCL shared time:", (end-start))

start = time.time()
logpdf_ocl = kde_normal.logpdf(test)
end = time.time()
print("OpenCL logtime:", (end-start))

start = time.time()
shared_logpdf_ocl = shared_kde_normal.logpdf(test)
end = time.time()
print("OpenCL shared logtime:", (end-start))

start = time.time()
pdf_truth = pdf_scipy.pdf(test.T)
end = time.time()
print("Python/Scipy time:", (end-start))

start = time.time()
logpdf_truth = pdf_scipy.logpdf(test.T)
end = time.time()
print("Python/Scipy log time:", (end-start))

print()
print("Comparing OpenCL with Python/Scipy results...")
print("pdf:")
compareResults(pdf_truth, pdf_ocl)
print("logpdf:")
compareResults(logpdf_truth, logpdf_ocl)

print("Shared pdf:")
compareResults(pdf_truth, shared_pdf_ocl)
print("Shared logpdf:")
compareResults(logpdf_truth, shared_logpdf_ocl)
