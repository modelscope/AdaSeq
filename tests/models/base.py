import hashlib
import os
import shutil
import unittest

from modelscope.utils.regress_test_utils import MsRegressTool, compare_arguments_nested


class TestModel(unittest.TestCase):
    def setUp(self):
        os.environ['REGRESSION_BASELINE'] = '1'
        self.is_baseline = (
            True if os.environ.get('IS_BASELINE', '').lower() in ['1', 'y', 'true'] else False
        )

        # RegressTool init
        regression_resource_path = os.path.abspath(os.path.join('tests', 'resources', 'regression'))

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
            baseline=self.is_baseline, store_func=store_func, load_func=load_func
        )


def compare_fn(value1, value2, key, type):
    # Ignore the differences between optimizers of two torch versions
    if type == 'cfg':
        return True
    if type != 'optimizer':
        return None

    match = value1['type'] == value2['type']
    shared_defaults = set(value1['defaults'].keys()).intersection(set(value2['defaults'].keys()))
    match = (
        all(
            [
                compare_arguments_nested(
                    f'Optimizer defaults {key} not match',
                    value1['defaults'][key],
                    value2['defaults'][key],
                )
                for key in shared_defaults
            ]
        )
        and match
    )
    match = (
        len(value1['state_dict']['param_groups']) == len(value2['state_dict']['param_groups'])
    ) and match
    for group1, group2 in zip(
        value1['state_dict']['param_groups'], value2['state_dict']['param_groups']
    ):
        shared_keys = set(group1.keys()).intersection(set(group2.keys()))
        match = (
            all(
                [
                    compare_arguments_nested(
                        f'Optimizer param_groups {key} not match', group1[key], group2[key]
                    )
                    for key in shared_keys
                ]
            )
            and match
        )
    return match
