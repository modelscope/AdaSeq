from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from uner.data.constant import (
    NON_ENTITY_LABEL,
    PAD_LABEL,
    PAD_LABEL_ID,
    PARTIAL_LABEL,
    PARTIAL_LABEL_ID,
)
from uner.metainfo import Preprocessors
from uner.utils.data_utils import gen_label2id_with_bio
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    module_name=Preprocessors.sequence_labeling_preprocessor)
class SequenceLabelingPreprocessor(NLPPreprocessor):

    def __init__(self,
                 model_dir: str,
                 labels: List[str] = None,
                 bio2bioes: bool = False,
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        label2id = kwargs.get('label2id', None)
        label2id = self.map_label_to_id(labels, label2id)
        self.label2id = transform_label2id(label2id, bio2bioes)
        self.bio2bioes = bio2bioes

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        output = super().__call__(data)
        if self.label2id is not None and isinstance(data,
                                                    Dict) and 'spans' in data:
            input_length = sum(output['emission_mask'])
            labels = spans_to_bio_labels(data['spans'], input_length)
            labels = transform_labels(labels, self.bio2bioes)
            output['label_ids'] = [
                self.label2id.get(labels[i], PARTIAL_LABEL_ID)
                for i in range(input_length)
            ]
        return output

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return gen_label2id_with_bio(labels)


def transform_label2id(label2id, bio2bioes=False):
    if bio2bioes:
        for label in sorted(label2id.keys()):
            if label[0] == 'B' and label.replace('B-', 'S-') not in label2id:
                label2id[label.replace('B-', 'S-')] = len(label2id)
            if label[0] == 'I' and label.replace('I-', 'E-') not in label2id:
                label2id[label.replace('I-', 'E-')] = len(label2id)
    return label2id


def transform_labels(labels, bio2bioes=False):
    if bio2bioes:
        new_labels = []
        for i, label in enumerate(labels):
            if label in [NON_ENTITY_LABEL, PARTIAL_LABEL]:
                new_labels.append(label)
            elif label[0] == 'B':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    new_labels.append(label)
                else:
                    new_labels.append(label.replace('B-', 'S-'))
            elif label[0] == 'I':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    new_labels.append(label)
                else:
                    new_labels.append(label.replace('I-', 'E-'))
            elif label[0] == 'S' or label[0] == 'E':
                new_labels.append(label)
            else:
                raise ValueError(f'Unrecognized label: {label}')
        return new_labels
    else:
        return labels


def spans_to_bio_labels(spans, length):
    labels = [NON_ENTITY_LABEL] * length
    for span in spans:
        if span['start'] < length:
            if span['type'] == PARTIAL_LABEL:
                labels[span['start']] = span['type']
            else:
                labels[span['start']] = 'B-' + span['type']
        for i in range(span['start'] + 1, span['end']):
            if i < length:
                if span['type'] == PARTIAL_LABEL:
                    labels[i] = span['type']
                else:
                    labels[i] = 'I-' + span['type']
    return labels
