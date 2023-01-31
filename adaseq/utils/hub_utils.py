import os

from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import Invoke


def get_or_download_model_dir(model, model_revision=None):
    """get model cache dir"""
    if os.path.exists(model):
        model_cache_dir = model if os.path.isdir(model) else os.path.dirname(model)
        check_local_model_is_latest(model_cache_dir, user_agent={Invoke.KEY: Invoke.LOCAL_TRAINER})
    else:
        model_cache_dir = snapshot_download(
            model, revision=model_revision, user_agent={Invoke.KEY: Invoke.TRAINER}
        )
    return model_cache_dir
