# Copyright (c) Alibaba, Inc. and its affiliates.
# Some codes are borrowed from AllenNLP.
# https://github.com/allenai/allennlp/blob/HEAD/allennlp/common/logging.py
# Copyright (c) AI2 AllenNLP. Licensed under the Apache License, Version 2.0.

import logging
import os
import sys
from logging import Filter
from os import PathLike
from typing import Union

from modelscope.utils.logger import get_logger


class AdaSeqLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps a set of messages to
    implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self._seen_msgs = set()

    def debug_once(self, msg, *args, **kwargs):  # noqa: D102
        if msg not in self._seen_msgs:
            self.debug(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def info_once(self, msg, *args, **kwargs):  # noqa: D102
        if msg not in self._seen_msgs:
            self.info(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def warning_once(self, msg, *args, **kwargs):  # noqa: D102
        if msg not in self._seen_msgs:
            self.warning(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def error_once(self, msg, *args, **kwargs):  # noqa: D102
        if msg not in self._seen_msgs:
            self.error(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def critical_once(self, msg, *args, **kwargs):  # noqa: D102
        if msg not in self._seen_msgs:
            self.critical(msg, *args, **kwargs)
            self._seen_msgs.add(msg)


logging.setLoggerClass(AdaSeqLogger)
logger = logging.getLogger(__name__)


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):  # noqa: D102
        return record.levelno < logging.ERROR


def prepare_global_logging(log_level: int = logging.INFO) -> None:
    """
    Prepare global logging.
    """

    # clear modelscop logger handlers to remove dumlicate logs.
    get_logger().handlers.clear()

    root_logger = logging.getLogger()

    # create handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    stderr_handler: logging.Handler = logging.StreamHandler(sys.stderr)

    for handler in [stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    if os.environ.get('ADASEQ_DEBUG'):
        LEVEL = logging.DEBUG
    else:
        LEVEL = log_level

    stdout_handler.setLevel(LEVEL)
    stdout_handler.addFilter(ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(LEVEL)

    # put all the handlers on the root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def prepare_logging(
    work_dir: Union[str, PathLike],
    rank: int = 0,
    world_size: int = 1,
    log_level: int = logging.INFO,
) -> None:
    """
    Prepare logging for training, log to file in `work_dir`.
    """
    root_logger = logging.getLogger()

    # create handlers
    if world_size == 1:
        log_file = os.path.join(work_dir, 'out.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    else:
        log_file = os.path.join(work_dir, f'out_worker{rank}.log')
        formatter = logging.Formatter(
            f'{rank} | %(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
    file_handler: logging.Handler = logging.FileHandler(log_file)
    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    stderr_handler: logging.Handler = logging.StreamHandler(sys.stderr)

    handler: logging.Handler
    for handler in [file_handler, stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    if os.environ.get('ADASEQ_DEBUG'):
        LEVEL = logging.DEBUG
    else:
        LEVEL = log_level

    file_handler.setLevel(LEVEL)
    stdout_handler.setLevel(LEVEL)
    stdout_handler.addFilter(ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(LEVEL)

    # put all the handlers on the root logger
    root_logger.addHandler(file_handler)
    if rank == 0:
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)
