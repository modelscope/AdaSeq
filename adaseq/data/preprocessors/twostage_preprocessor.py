# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from .pretraining_preprocessor import PretrainingPreprocessor


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.twostage_preprocessor)
class TwoStagePreprocessor(PretrainingPreprocessor):
    """Preprocessor for twostage-ner.
    span targets are processed into mention_boundary, type_ids, ident_ids.
    examples:
        span: {'start':1, 'end':2, 'type': ['PER']}
        processed: {'mention_boundary': [[1], [2]], 'type_ids':[1], 'ident_ids': ['S-SPAN']]}
    """

    def _generate_mentions_labels(self, data: Union[str, List, Dict], output: Dict[str, Any]):
        mention_type_ids = []
        boundary_starts = []
        boundary_ends = []
        mention_mask = []
        for span in data['spans']:
            boundary_starts.append(span['start'])
            boundary_ends.append(span['end'] - 1)
            mention_mask.append(1)
            mention_type_ids.append(self.typing_label_to_id.get(span['type'], -1))

        output['mention_boundary'] = [boundary_starts, boundary_ends]
        output['mention_mask'] = mention_mask
        output['type_ids'] = mention_type_ids
