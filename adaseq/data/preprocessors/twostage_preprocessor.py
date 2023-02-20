# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.twostage_preprocessor)
class TwoStagePreprocessor(NLPPreprocessor):
    """Preprocessor for twostage-ner.
    span targets are processed into mention_boundary, type_ids, ident_ids.
    examples:
        span: {'start':1, 'end':2, 'type': ['PER']}
        processed: {'mention_boundary': [[1], [2]], 'type_ids':[1], 'ident_ids': ['S-SPAN']]}
    """

    def __init__(self, model_dir: str, **kwargs) -> None:

        super().__init__(model_dir, return_offsets=True, **kwargs)

        label_to_id = kwargs.pop('label_to_id', None)
        labels = kwargs.pop('labels', None)
        if label_to_id is not None:
            self.typing_label_to_id = label_to_id
        elif labels is not None:
            self.typing_label_to_id = self.make_label_to_id(labels)
        else:
            raise ValueError('Must have one of `labels` or `label_to_id`')
        self.typing_id_to_label = {
            v: k for k, v in sorted(self.typing_label_to_id.items(), key=lambda x: x[1])
        }
        # make sure they are aligned.
        assert len(self.typing_id_to_label) not in self.typing_id_to_label
        assert len(self.typing_id_to_label) - 1 in self.typing_id_to_label

        self.ident_label_to_id = {'O': 0, 'B-SPAN': 1, 'I-SPAN': 2, 'E-SPAN': 3, 'S-SPAN': 4}
        self.ident_id_to_label = {0: 'O', 1: 'B-SPAN', 2: 'I-SPAN', 3: 'E-SPAN', 4: 'S-SPAN'}

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for two-stage-ner model"""
        output = super().__call__(data)
        if isinstance(data, Dict):
            # span detection inputs
            output['ident_ids'] = self.__span2bioes(
                len(output['tokens']['mask']) - 2, data['spans']
            )
            # entity typing inputs
            self._generate_mentions_labels(data, output)
        return output

    def _generate_mentions_labels(self, data: Union[str, List, Dict], output: Dict[str, Any]):
        mention_type_ids = []
        boundary_starts = []
        boundary_ends = []
        mention_mask = []
        for span in data['spans']:
            if (
                span['start'] > len(output['tokens']['mask']) - 2
                or span['end'] > len(output['tokens']['mask']) - 2
            ):
                continue
            boundary_starts.append(span['start'])
            boundary_ends.append(span['end'] - 1)
            mention_mask.append(1)
            mention_type_ids.append(self.typing_label_to_id.get(span['type'], -1))

        output['mention_boundary'] = [boundary_starts, boundary_ends]
        output['mention_mask'] = mention_mask
        output['type_ids'] = mention_type_ids

    def __span2bioes(self, sequence_length: int, spans: List[Dict]) -> List[int]:
        ident_ids = [self.ident_label_to_id['O']] * sequence_length
        for span in spans:
            start = span['start']
            end = span['end']
            if start > sequence_length or end > sequence_length:
                continue
            if end - start == 1:
                ident_ids[start] = self.ident_label_to_id['S-SPAN']
            else:
                for i in range(start, end):
                    ident_ids[i] = self.ident_label_to_id['I-SPAN']
                ident_ids[start] = self.ident_label_to_id['B-SPAN']
                ident_ids[end - 1] = self.ident_label_to_id['E-SPAN']
        return ident_ids
