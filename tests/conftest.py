def pytest_addoption(parser):
    parser.addoption('--times', action='store_true', dest="times",
                     default=False, help="enable all time tests")
    parser.addoption('--times-ocl', action='store_true', dest="times_ocl",
                     default=False, help="enable all OpenCL time tests")
    parser.addoption('--times-scipy', action='store_true', dest="times_scipy",
                     default=False, help="enable all Scipy time tests")

def pytest_configure(config):
    setattr(config.option, 'markexpr', 'not times')
    if config.option.times:
        setattr(config.option, 'markexpr', 'times')
    if config.option.times_ocl:
        setattr(config.option, 'markexpr', 'times_ocl')
    if config.option.times_scipy:
        setattr(config.option, 'markexpr', 'times_scipy')