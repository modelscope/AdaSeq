# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from adaseq.metainfo import Preprocessors
from ..constant import NONE_REL_LABEL, NONE_REL_LABEL_ID
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(module_name=Preprocessors.relation_extraction_preprocessor)
class RelationExtractionPreprocessor(NLPPreprocessor):
    """ Relation Extraction data preprocessor """

    def __init__(self, model_dir: str, labels: List[str] = None, **kwargs):
        super().__init__(model_dir, return_emission_mask=True, **kwargs)

        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """ prepare inputs for Relation Extraction model. """
        output = super().__call__(data)

        if self.label2id is not None and isinstance(data, Dict) and 'label' in data:
            label = data['label']
            output['label_id'] = [self.label2id[label]]
        output['so_head_mask'] = data['so_head_mask']
        return output

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return self._gen_label2id(labels)

    @staticmethod
    def _gen_label2id(labels: List[str]) -> Dict[str, int]:
        label2id = {}
        label2id[NONE_REL_LABEL] = NONE_REL_LABEL_ID
        for label in labels:
            if label != NONE_REL_LABEL:
                label2id[f'{label}'] = len(label2id)
        return label2id
