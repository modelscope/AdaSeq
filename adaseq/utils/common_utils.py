# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
from typing import Dict

from modelscope.utils.config import Config


def create_datetime_str() -> str:
    """Create a string indicating current time

    Create a string indicating current time in microsecond precision,
    for example, 221109144626.861616

    Returns:
        str: current time string
    """
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime('%y%m%d%H%M%S.%f')
    return datetime_str


def has_keys(_dict: Dict, *keys: str):
    """Check whether a nested dict has a key

    Args:
        _dict (Dict): a nested dict like object
        *keys (str): flattened key list

    Returns:
        bool: whether _dict has keys
    """
    if not _dict or not keys:
        return False

    sub_dict = _dict
    for key in keys:
        if isinstance(sub_dict, (dict, Config)) and key in sub_dict:
            sub_dict = sub_dict[key]
        else:
            return False
    return True
