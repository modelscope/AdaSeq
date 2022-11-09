import datetime
import os.path as osp

from datasets.load import dataset_module_factory, import_main_class
from modelscope.utils.config import Config


# copy from  hunggingface datasets load.py load_dataset_builder
def import_dataset_builder_class(builder_name):
    builder_path = osp.join(
        osp.dirname(osp.dirname(osp.realpath(__file__))), 'data',
        'dataset_builders', f'{builder_name}.py'
    )  # maybe failed where working dir does not has uer dir.
    dataset_module = dataset_module_factory(builder_path)
    # Get dataset builder class from the processing script
    builder_cls = import_main_class(dataset_module.module_path)
    return builder_cls


def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime('%y%m%d%H%M%S.%f')
    return datetime_str


def has_keys(_dict, *keys):
    if not _dict or not keys:
        return False

    sub_dict = _dict
    for key in keys:
        if isinstance(sub_dict, (dict, Config)) and key in sub_dict:
            sub_dict = sub_dict[key]
        else:
            return False
    return True
