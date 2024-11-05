import logging

disable_loggers = ["solib", "perscache"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.disabled = True

def pytest_addoption(parser):
    parser.addoption('--runhf', action='store_true', default=False,
                     help='Run tests with slow HF models')
