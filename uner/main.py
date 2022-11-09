#!/usr/bin/env python
import os
import sys

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def run():
    from uner.commands import main
    main(prog='adaseq')


if __name__ == '__main__':
    run()
