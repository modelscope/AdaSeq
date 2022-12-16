# Copyright (c) Alibaba, Inc. and its affiliates.
import os


def is_empty_dir(path):
    """Check if a directory is empty"""
    return len(list(filter(lambda x: not x.startswith('.nfs'), os.listdir(path)))) == 0
