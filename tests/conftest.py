def pytest_addoption(parser):
    parser.addoption('--benchmark', action='store_true', dest="benchmark",
                     default=False, help="enable all benchmark tests")

def pytest_configure(config):
    if not config.option.benchmark:
        setattr(config.option, 'markexpr', 'not benchmark')