import numpy as np
from six import string_types
import atexit

from ._ffi import ffi, lib

class CFFIDoubleArray(object):
    def __init__(self, array, ffi):
        self.shape = ffi.new("size_t[]", array.shape)
        self.strides = ffi.new("size_t[]", array.strides)
        self.arrayptr = ffi.cast("double*", array.ctypes.data)
        self.cffiarray = ffi.new('DoubleNumpyArray*', {'ptr': self.arrayptr,
                                                              'size': array.size,
                                                              'ndim': array.ndim,
                                                              'shape': self.shape,
                                                              'strides': self.strides})
    def c_ptr(self):
        return self.cffiarray

class Error:
    NoError = 0
    MemoryError = 1

class gaussian_kde_ocl(object):

    default_pro_que = None

    def __init__(self, dataset, bw_method=None, pro_que=None):
        """
        Creates a gaussian_kde_ocl model. The dataset should be a two dimensional Numpy array **stored in row-major
        order**. There is no compatibility, for now, with general Numpy arrays. Use numpy.ascontiguosarray to obtain
        a row-major order array.
        :param dataset: Train data with n instances of d dimensions and shape (n,d).
        :param bw_method: Bandwidth computation method.
        """
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        if self.dataset.ndim != 2:
            raise ValueError("Dataset matrix should have a maximum of 2 dimensions . "
                             "Dataset dimensions:" + self.dataset.ndim)
        if self.dataset.dtype != np.float64:
            self.dataset = self.dataset.astype(np.float64)

        if self.dataset.shape[0] == 1:
            self.dataset = self.dataset.T

        self.n, self.d = self.dataset.shape
        self.set_bandwidth(bw_method=bw_method)
        self._initCFFI(pro_que)

    def scotts_factor(self):
        return np.power(self.n, -1. / (self.d + 4))

    def silverman_factor(self):
        return np.power(self.n * (self.d + 2.0) / 4.0, -1. / (self.d + 4))

    def set_bandwidth(self, bw_method=None):
        """
        Compute bandwidth
        :param bw_method: Method to compute bandwidth, as in scipy.stats.gaussian_kde.
        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=False,
                                                         bias=False))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = (self._data_covariance * self.factor**2)
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) * self.n
        self.cholesky = np.linalg.cholesky(self.covariance)


    def _initCFFI(self, pro_que=None):
        """
        Initializes all the CFFI code
        :return:
        """
        if pro_que is None:
            if gaussian_kde_ocl.default_pro_que is None:
                gaussian_kde_ocl.default_pro_que = lib.new_proque()
                atexit.register(lib.gaussian_proque_free, gaussian_kde_ocl.default_pro_que)

            self.pro_que = gaussian_kde_ocl.default_pro_que
        else:
            self.pro_que = pro_que

        chol = CFFIDoubleArray(self.cholesky, ffi)
        dataset = CFFIDoubleArray(self.dataset, ffi)
        error = ffi.new("Error*", 0)
        self.kdedensity = lib.gaussian_kde_init(self.pro_que, chol.c_ptr(), dataset.c_ptr(), error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        self.kdedensity = ffi.gc(self.kdedensity, lib.gaussian_kde_free)

    def evaluate(self, points):
        """
        Evaluates the probability density function (pdf) of a set of m instances with d dimensions. The points array should
        be row-major order array with shape (m,d)
        :param points: Test instances with shape (m, d).
        :return out: ndarray with shape (m,) with the pdf of each point.
        """
        self.pdf(points)

    def pdf(self, x):
        """
        See evaluate.
        :param x: Test instances with shape (m, d).
        :return out: ndarray with shape (m,) with the pdf of each point.
        """
        points = np.atleast_2d(x)
        m, d = points.shape
        if d != self.d:
            if m == 1 == self.d:
                # points was passed in as a row vector
                points = points.T
                m = d
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        result = np.empty((m,), dtype=np.float64)
        cffi_points = CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", 0)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        lib.gaussian_kde_pdf(self.kdedensity, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        return result

    __call__ = pdf

    def logpdf(self, x):
        """
        Evaluates the log probability density function (logpdf) of a set of m instances with d dimensions. The x array
        should be row-major order array with shape (m,d).
        :param x: Test instances with shape (m, d).
        :return out: ndarray with shape (m,) with the pdf of each point.
        """
        points = np.atleast_2d(x)
        m, d = points.shape
        if d != self.d:
            if m == 1 == self.d:
                # points was passed in as a row vector
                points = points.T
                m = d
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        result = np.empty((m,), dtype=np.float64)
        cffi_points = CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", 0)
        lib.gaussian_kde_logpdf(self.kdedensity, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        return result

    def new_proque(self):
        return lib.new_proque()