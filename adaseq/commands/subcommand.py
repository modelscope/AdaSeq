# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
from abc import ABC, abstractmethod


class Subcommand(ABC):
    """Abstract class for subcommands"""

    @classmethod
    @abstractmethod
    def add_subparser(cls, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add arguments parser for subcommand"""
        pass
