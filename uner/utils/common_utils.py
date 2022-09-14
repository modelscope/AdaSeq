import datetime

from modelscope.utils.config import Config


def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%y%m%d%H%M%S.%f")
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
