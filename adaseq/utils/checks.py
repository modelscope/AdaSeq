# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Functions and exceptions for checking that
AdaSeq and its models are configured correctly.
"""
from typing import Any, Tuple, Union


# Copyright (c) AI2 AllenNLP. Licensed under the Apache License, Version 2.0.
class ConfigurationError(Exception):
    """
    The exception raised by any Adaseq object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return type(self), (self.message,)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message
