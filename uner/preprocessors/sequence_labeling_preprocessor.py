from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from .constant import NON_ENTITY_LABEL,  PAD_LABEL, PAD_LABEL_ID
from .nlp_preprocessor import NLPPreprocessor
from uner.metainfo import Preprocessors


@PREPROCESSORS.register_module(module_name=Preprocessors.sequence_labeling_preprocessor)
class SequenceLabelingPreprocessor(NLPPreprocessor):
    def __init__(self, model_dir: str, label2id, bio2bioes=False, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.label2id = self.transform_label2id(label2id, bio2bioes)
        self.bio2bioes = bio2bioes

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        output = super().__call__(data)
        if self.label2id is not None and isinstance(data, Dict) and 'labels' in data:
            input_length = sum(output['emission_mask'])
            labels = self.transform_labels(data['labels'], self.bio2bioes)
            output['label_ids'] = [self.label2id[labels[i]] for i in range(input_length)]
        return output

    @staticmethod
    def transform_label2id(label2id, bio2bioes=False):
        if bio2bioes:
            for label in sorted(label2id.keys()):
                if label[0] == 'B' and label.replace('B-', 'S-') not in label2id:
                    label2id[label.replace('B-', 'S-')] = len(label2id)
                if label[0] == 'I' and label.replace('I-', 'E-') not in label2id:
                    label2id[label.replace('I-', 'E-')] = len(label2id)
        return label2id

    @staticmethod
    def transform_labels(labels, bio2bioes=False):
        if bio2bioes:
            new_labels = []
            for i, label in enumerate(labels):
                if label in [NON_ENTITY_LABEL, PAD_LABEL]:
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

