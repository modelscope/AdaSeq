# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict, Optional

from modelscope.metrics.base import Metric


class DatasetDumper(Metric):
    """class to dump model predictions"""

    def __init__(self, save_path: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.data = []

    def add(self, outputs: Dict, inputs: Dict):
        """add predictions to cache"""
        raise NotImplementedError

    def evaluate(self):
        """creat dump file and dump predicitons."""
        if self.save_path is None:
            self.save_path = os.path.join(self.trainer.work_dir, 'pred.txt')
        save_path = self.save_path
        self.save_path = save_path + '.tmp'  # file lock
        self.dump()
        os.replace(self.save_path, save_path)
        self.save_path = save_path
        return {}

    def dump(self):
        """dump predictions"""
        raise NotImplementedError

    def merge(self):
        """implement abstract method
        This function should never be used.
        """
        raise NotImplementedError
