import os
import unittest

from modelscope.trainers import build_trainer
from modelscope.utils.regress_test_utils import MsRegressTool

from uner.metainfo import Trainers
from uner.metrics import *  # noqa
from uner.preprocessors import *  # noqa
from uner.trainers import *  # noqa
from uner.trainers.hooks import *  # noqa


class TestRegistry(unittest.TestCase):

    def setUp(self):
        os.environ['REGRESSION_BASELINE'] = '1'
        self.is_baseline = True if os.environ.get(
            'IS_BASELINE', '').lower() in ['1', 'y', 'true'] else False

    def test_bert_softmax(self):
        pass

    def test_bert_crf(self):
        regress_tool = MsRegressTool(baseline=self.is_baseline)

        # def cfg_modify_fn(cfg):
        #     cfg.work_dir = 'tmp'
        #     cfg.task = 'named-entity-recognition'
        #     cfg.dataset = {'corpus': 'resume'}
        #     cfg.preprocessor = {
        #         'type': 'sequence-labeling-preprocessor',
        #         'model_dir': 'quincyqiang/nezha-cn-base',
        #         'max_length': 150,
        #         'bio2bioes': True
        #     }
        #     cfg.model = {
        #         'type': 'sequence-labeling-model',
        #         'encoder': {
        #             'type': 'nezha',
        #             'model_name_or_path': 'quincyqiang/nezha-cn-base'
        #         },
        #         'word_dropout': 0.1,
        #         'use_crf': true
        #     }
        #     cfg.train.max_epochs = 3
        #     cfg.train.optimizer = {
        #         'type': 'AdamW',
        #         'lr': 5.0e-5,
        #         'crf_lr': 5.0e-1
        #     }
        #     cfg.train.lr_scheduler = {
        #         'type': 'LinearLR',
        #         'start_factor': 1.0,
        #         'end_factor': 0.0,
        #         'total_iters': 20
        #     }
        #     return cfg
        cfg_file = 'tests/resources/configs/train_bert_crf.yaml'
        trainer = build_trainer(
            Trainers.ner_trainer, 
            default_args={'cfg_file': cfg_file, 'seed': 42})

        with regress_tool.monitor_ms_train(trainer, 'ut_bert_crf.bin', level='strict'):
            trainer.train()
            


if __name__ == '__main__':
    unittest.main()
