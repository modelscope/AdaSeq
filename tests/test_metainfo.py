import unittest

from modelscope.metainfo import Hooks as MsHooks
from modelscope.metainfo import Metrics as MsMetrics
from modelscope.metainfo import Models as MsModels
from modelscope.metainfo import Pipelines as MsPipelines
from modelscope.metainfo import Preprocessors as MsPreprocessors
from modelscope.metainfo import Trainers as MsTrainers
from modelscope.utils.constant import Tasks as MsTasks

from adaseq.metainfo import (
    Hooks,
    Metrics,
    Models,
    Pipelines,
    Preprocessors,
    Tasks,
    Trainers,
    get_member_set,
)


class TestMetaInfo(unittest.TestCase):
    def test_tasks(self):
        """Make sure all AdaSeq task names are the same as ModelScope"""
        adaseq_tasks = get_member_set(Tasks)
        modelscope_tasks = get_member_set(MsTasks)
        for task in adaseq_tasks:
            if task not in modelscope_tasks:
                print(f'\n[WARNING] AdaSeq task "{task}" not in Modelscope Task!')

    def test_pipelines(self):
        """Make sure all AdaSeq pipeline names are different from Modelscope"""
        adaseq_pipelines = get_member_set(Pipelines)
        modelscope_pipelines = get_member_set(MsPipelines)
        self.assertEqual(adaseq_pipelines & modelscope_pipelines, set())

    def test_models(self):
        """Make sure all AdaSeq model names are different from Modelscope"""
        adaseq_models = get_member_set(Models)
        modelscope_models = get_member_set(MsModels)
        self.assertEqual(adaseq_models & modelscope_models, set())

    def test_preprocessors(self):
        """Make sure all AdaSeq preprocessor names are different from Modelscope"""
        adaseq_preprocessors = get_member_set(Preprocessors)
        modelscope_preprocessors = get_member_set(MsPreprocessors)
        self.assertEqual(adaseq_preprocessors & modelscope_preprocessors, set())

    def test_trainers(self):
        """Make sure all AdaSeq trainer names are different from Modelscope"""
        adaseq_trainers = get_member_set(Trainers)
        modelscope_trainers = get_member_set(MsTrainers)
        self.assertEqual(adaseq_trainers & modelscope_trainers, set())

    def test_metrics(self):
        """Make sure all AdaSeq metric names are different from Modelscope"""
        adaseq_metrics = get_member_set(Metrics)
        modelscope_metrics = get_member_set(MsMetrics)
        self.assertEqual(adaseq_metrics & modelscope_metrics, set())

    def test_hooks(self):
        """Make sure all AdaSeq hook names are different from Modelscope"""
        adaseq_hooks = get_member_set(Hooks)
        modelscope_hooks = get_member_set(MsHooks)
        self.assertEqual(adaseq_hooks & modelscope_hooks, set())


if __name__ == '__main__':
    unittest.main()
