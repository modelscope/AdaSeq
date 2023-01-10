# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify

from adaseq.metainfo import Pipelines, Tasks

from .base import Pipeline


@PIPELINES.register_module(
    Tasks.named_entity_recognition, module_name=Pipelines.span_based_ner_pipeline
)
class SpanBasedNERPipeline(Pipeline):
    """
    A NER pipeline for span based models, e.g. biaffine model and global pointer model.
    Model outputs should contain:
        predicts: [[(start, end, type), ...]]
    Pipeline outputs will be like:
        {"output": [{"span": "国正", "type": "PER", "start": 0, "end": 2}, ...]}
    """

    def postprocess(  # noqa: D102
        self, inputs: Dict[str, Any], **postprocess_params
    ) -> Dict[str, Any]:
        text = inputs['meta']['text']

        # TODO post_process does not support batch for now.
        offset_mapping = inputs['tokens']['offset_mapping']
        if offset_mapping.dim() == 3:
            offset_mapping = offset_mapping[0]
        offset_mapping = torch_nested_numpify(torch_nested_detach(offset_mapping))

        chunks = inputs['predicts']  # chunks: [[(start, end, type), ...]]
        if isinstance(chunks[0], list):
            chunks = chunks[0]
        chunks = sorted(chunks, key=lambda x: x[0])

        outputs = []
        for chunk in chunks:
            start = offset_mapping[chunk[0]][0]
            end = offset_mapping[chunk[1] - 1][1]
            span_type = chunk[2]
            outputs.append({'span': text[start:end], 'type': span_type, 'start': start, 'end': end})

        return {OutputKeys.OUTPUT: outputs}
