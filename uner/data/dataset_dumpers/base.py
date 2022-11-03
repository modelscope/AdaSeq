import os
from typing import Dict

from modelscope.metrics.base import Metric


class DatasetDumper(Metric):

    def __init__(self, save_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.data = []

    def add(self, outputs: Dict, inputs: Dict):
        raise NotImplementedError

    def evaluate(self):
        if self.save_path is None:
            self.save_path = os.path.join(self.trainer.work_dir, 'test.txt')
        save_path = self.save_path
        self.save_path = save_path + '.tmp'  # file lock
        self.dump()
        os.replace(self.save_path, save_path)
        self.save_path = save_path
        return {}

    def dump(self):
        raise NotImplementedError
