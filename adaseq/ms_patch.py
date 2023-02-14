# Copyright (c) Alibaba, Inc. and its affiliates.
# Fix some modelscope bugs temporarily.
# These bugs will be fixed in the next modelscope version.


def suppress_modelscope_ast_warning():  # noqa
    try:
        from modelscope.utils.logger import get_logger

        def filter_modelscope_ast_warning(record):
            return 'not found in ast index file' not in record.msg

        logger = get_logger()
        logger.addFilter(filter_modelscope_ast_warning)
    except IsADirectoryError:
        pass


suppress_modelscope_ast_warning()
