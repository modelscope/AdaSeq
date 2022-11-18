#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def run():
    """Command line main interface"""
    from adaseq.commands import main

    main(prog='adaseq')


if __name__ == '__main__':
    run()
