"""
Helper methods for logging
"""
import os
import logging


def configure_logger(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_dir = os.environ['LOG_DIR']

    fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(name)))
    fh.setLevel(level)
    fh_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    fh.setFormatter(fh_formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(level)
    ch_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
