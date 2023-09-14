# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify

from adaseq.metainfo import Pipelines, Tasks

from .base import Pipeline


@PIPELINES.register_module(
    Tasks.named_entity_recognition, module_name=Pipelines.sequence_labeling_pipeline
)
class SequenceLabelingPipeline(Pipeline):
    """
    A pipeline for sequence labeling models
    Model outputs should contain:
        predicts: [[0, 0, 1, 2, 2, ...]]
    Pipeline outputs will be like:
        {"output": [{"span": "国正", "type": "PER", "start": 0, "end": 2}, ...]}
    """

    def postprocess(  # noqa: D102
        self, inputs: Dict[str, Any], **postprocess_params
    ) -> Dict[str, Any]:
        text = inputs['meta']['text']

        # TODO post_process does not support batch for now.
        predictions = inputs['predicts']  # predicts: [[0, 0, 1, 2, 2, ...]]
        if len(predictions.shape) == 2:
            predictions = predictions[0]
        predictions = torch_nested_numpify(torch_nested_detach(predictions))

        offset_mapping = inputs['tokens']['offset_mapping']
        if offset_mapping.dim() == 3:
            offset_mapping = offset_mapping[0]
        offset_mapping = torch_nested_numpify(torch_nested_detach(offset_mapping))

        labels = [self.id2label[x] for x in predictions]

        return_prob = postprocess_params.pop('return_prob', True)
        if return_prob:
            if 'logits' in inputs:
                logits = inputs['logits']
                if len(logits.shape) == 3:
                    logits = logits[0]
                probs = torch_nested_numpify(torch_nested_detach(logits.softmax(-1)))
            else:
                return_prob = False

        outputs = []
        chunk = {}
        for i, (label, offsets) in enumerate(zip(labels, offset_mapping)):
            if label[0] in 'BS':
                if chunk:
                    chunk['span'] = text[chunk['start'] : chunk['end']]
                    outputs.append(chunk)
                chunk = {}

            if label[0] in 'BIES':
                if not chunk:
                    chunk = {'type': label[2:], 'start': offsets[0], 'end': offsets[1]}
                    if return_prob:
                        chunk['prob'] = probs[i][predictions[i]]

            if label[0] in 'IES':
                if chunk:
                    chunk['end'] = offsets[1]

            if label[0] in 'ES':
                if chunk:
                    chunk['span'] = text[chunk['start'] : chunk['end']]
                    outputs.append(chunk)
                    chunk = {}

        if chunk:
            chunk['span'] = text[chunk['start'] : chunk['end']]
            outputs.append(chunk)

        return {OutputKeys.OUTPUT: outputs}
