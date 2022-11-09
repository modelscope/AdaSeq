import argparse
from abc import ABC, abstractmethod


class Subcommand(ABC):

    @classmethod
    @abstractmethod
    def add_subparser(
            cls,
            parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        pass
