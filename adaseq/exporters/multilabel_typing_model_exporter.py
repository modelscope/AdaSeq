from collections import OrderedDict
from typing import Any, Dict, Mapping

from modelscope.exporters.builder import EXPORTERS
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModeKeys

from adaseq.metainfo import Models, Tasks

from .base import Exporter


@EXPORTERS.register_module(Tasks.entity_typing, module_name=Models.multilabel_span_typing_model)
class MultiLabelSpanTypingModelExporter(Exporter):
    """An exporter for span based multilabel entity typing  model"""

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:  # noqa: D102
        if self.preprocessor is not None:
            preprocessor = self.preprocessor
            preprocessor.mode = ModeKeys.INFERENCE
        else:
            assert hasattr(
                self.model, 'model_dir'
            ), 'model_dir attribute is required to build the preprocessor'
            preprocessor = Preprocessor.from_pretrained(self.model.model_dir)
        dummy_inputs = preprocessor(
            {'tokens': ['小', '明', '爱', '学', '习'], 'spans': [{'start': 1, 'end': 2}]}
        )
        print(dummy_inputs)
        return dummy_inputs

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:  # noqa: D102
        return OrderedDict(
            [
                ('input_ids', {0: 'batch', 1: 'sequence'}),
                ('attention_mask', {0: 'batch', 1: 'sequence'}),
                ('has_special_token', {0: 'batch'}),
                ('offsets', {0: 'batch', 1: 'orig_sequence'}),
                ('mask', {0: 'batch', 1: 'orig_sequence'}),
                ('mention_boundary', {0: 'batch', 1: 'offset_type', 2: 'offset'}),
                ('mention_mask', {0: 'batch', 1: 'mentions'}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:  # noqa: D102
        return OrderedDict(
            [
                ('logits', {0: 'batch', 1: 'mentions', 2: 'logits'}),
                ('predicts', {0: 'batch', 1: 'mention', 2: 'types'}),
            ]
        )
