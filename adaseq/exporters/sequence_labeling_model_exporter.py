from collections import OrderedDict
from typing import Any, Dict, Mapping

from modelscope.exporters.builder import EXPORTERS
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import ModeKeys

from adaseq.metainfo import Models, Tasks

from .base import Exporter


@EXPORTERS.register_module(Tasks.word_segmentation, module_name=Models.sequence_labeling_model)
@EXPORTERS.register_module(Tasks.part_of_speech, module_name=Models.sequence_labeling_model)
@EXPORTERS.register_module(
    Tasks.named_entity_recognition, module_name=Models.sequence_labeling_model
)
class SequenceLabelingModelExporter(Exporter):
    """An exporter for sequence labeling model"""

    def generate_dummy_inputs(self, **kwargs) -> Dict[str, Any]:  # noqa: D102
        if self.preprocessor is not None:
            preprocessor = self.preprocessor
            preprocessor.mode = ModeKeys.INFERENCE
        else:
            assert hasattr(
                self.model, 'model_dir'
            ), 'model_dir attribute is required to build the preprocessor'
            preprocessor = Preprocessor.from_pretrained(self.model.model_dir)
        dummy_inputs = preprocessor('2023')
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
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:  # noqa: D102
        return OrderedDict(
            [
                ('predicts', {0: 'batch', 1: 'orig_sequence'}),
            ]
        )
