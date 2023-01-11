# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from ..constant import NONE_REL_LABEL, NONE_REL_LABEL_ID
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.relation_extraction_preprocessor
)
class RelationExtractionPreprocessor(NLPPreprocessor):
    """Relation Extraction data preprocessor"""

    def __init__(self, model_dir: str, labels: List[str], **kwargs):
        label_to_id = self._gen_label2id(labels)
        super().__init__(model_dir, label_to_id=label_to_id, return_offsets=True, **kwargs)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Relation Extraction model."""
        output = super().__call__(data)

        if isinstance(data, Dict) and 'label' in data:
            output['label_id'] = self.label_to_id[data['label']]
        output['so_head_mask'] = data['so_head_mask']
        return output

    @staticmethod
    def _gen_label2id(labels: List[str]) -> Dict[str, int]:
        label2id = {}
        label2id[NONE_REL_LABEL] = NONE_REL_LABEL_ID
        for label in labels:
            if label != NONE_REL_LABEL:
                label2id[f'{label}'] = len(label2id)
        return label2id
