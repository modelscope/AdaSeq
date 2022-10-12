import hashlib
import os
import shutil
import unittest

from modelscope.trainers import build_trainer
from tests.regress_test_utils import MsRegressTool

from uner.metainfo import Trainers


class TestRegistry(unittest.TestCase):

    def setUp(self):
        os.environ['REGRESSION_BASELINE'] = '1'
        self.is_baseline = True if os.environ.get(
            'IS_BASELINE', '').lower() in ['1', 'y', 'true'] else False

        # fix modelscope bug
        from modelscope.trainers import EpochBasedTrainer
        from modelscope.utils import regress_test_utils
        from tests.ms_patch import train_step, numpify_tensor_nested
        EpochBasedTrainer.train_step = train_step
        regress_test_utils.numpify_tensor_nested = numpify_tensor_nested

        # RegressTool init
        regression_resource_path = os.path.abspath(
            os.path.join('tests', 'resources', 'regression'))

        def store_func(local, remote):
            os.makedirs(regression_resource_path, exist_ok=True)
            shutil.copy(local, os.path.join(regression_resource_path, remote))

        def load_func(local, remote):
            baseline = os.path.join(regression_resource_path, remote)
            if not os.path.exists(baseline):
                raise ValueError(f'base line file {baseline} not exist')
            print(
                f'local file found:{baseline}, md5:{hashlib.md5(open(baseline,"rb").read()).hexdigest()}'
            )
            if os.path.exists(local):
                os.remove(local)
            os.symlink(baseline, local, target_is_directory=False)

        self.regress_tool = MsRegressTool(
            baseline=self.is_baseline,
            store_func=store_func,
            load_func=load_func)

    def test_bert_softmax(self):
        pass

    def test_bert_crf(self):
        cfg_file = os.path.join('tests', 'resources', 'configs',
                                'train_bert_crf.yaml')
        trainer = build_trainer(
            Trainers.ner_trainer,
            default_args={
                'cfg_file': cfg_file,
                'seed': 42
            })

        with self.regress_tool.monitor_ms_train(
                trainer, 'ut_bert_crf', level='strict', atol=1e-4):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
