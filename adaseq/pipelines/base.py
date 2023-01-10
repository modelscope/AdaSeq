# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline as MsPipeline
from modelscope.preprocessors import Preprocessor


class Pipeline(MsPipeline):
    """use `model` and `preprocessor` to create a pipeline for prediction

    Args:
        model (str or Model): A model instance or a model local dir or a model id in the model hub.
        preprocessor (Preprocessor): a preprocessor instance, must not be None.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    """

    def __init__(
        self,
        model: Union[Model, str],
        preprocessor: Optional[Preprocessor] = None,
        config_file: str = None,
        device: str = 'gpu',
        auto_collate: bool = True,
        **kwargs
    ):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
        )

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(self.model.model_dir, **kwargs)
        self.model.eval()

        assert hasattr(self.preprocessor, 'id_to_label')
        self.id2label = self.preprocessor.id_to_label

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:  # noqa: D102
        with torch.no_grad():
            return {**self.model(**inputs, **forward_params), **inputs}

    def postprocess(  # noqa: D102
        self, inputs: Dict[str, Any], **postprocess_params
    ) -> Dict[str, Any]:
        raise NotImplementedError
