# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import Dict

from modelscope.metrics.builder import METRICS
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import DatasetDumpers

from .base import DatasetDumper


@METRICS.register_module(module_name=DatasetDumpers.ner_dumper)
class NamedEntityRecognitionDatasetDumper(DatasetDumper):
    """Named Entity Recognition dumper."""

    def __init__(self, model_type: str, dump_format: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.dump_format = dump_format

    def add(self, outputs: Dict, inputs: Dict):
        """Only support sequence_labeling models now."""
        if self.model_type == 'sequence_labeling':
            self._add_sequence_labeling_data(outputs, inputs)
        elif self.model_type == 'span_based':
            self._add_span_based_data(outputs, inputs)
        else:
            raise NotImplementedError

    def dump(self):
        """Only support dump CoNLL format and jsonline now."""
        if self.dump_format == 'conll':
            self._dump_to_conll()
        elif self.dump_format == 'jsonline':
            self._dump_to_jsonline()
        else:
            raise NotImplementedError

    def _add_sequence_labeling_data(self, outputs: Dict, inputs: Dict):
        if is_parallel(self.trainer.model):
            id2label = self.trainer.model.module.id_to_label
        else:
            id2label = self.trainer.model.id_to_label
        batch_meta = inputs['meta']
        batch_labels = torch_nested_numpify(torch_nested_detach(inputs['label_ids'])).tolist()
        batch_predicts = torch_nested_numpify(torch_nested_detach(outputs['predicts'])).tolist()
        for meta, labels, predicts in zip(batch_meta, batch_labels, batch_predicts):
            self.data.append(
                {
                    'tokens': meta['tokens'],
                    'labels': [id2label[x] for x in labels if x != PAD_LABEL_ID],
                    'predicts': [id2label[x] for x in predicts if x != PAD_LABEL_ID],
                }
            )

    def _add_span_based_data(self, outputs: Dict, inputs: Dict):
        for meta, predicts in zip(inputs['meta'], outputs['predicts']):
            obj = dict(tokens=meta['tokens'], spans=meta['spans'], predicts=predicts)
            self.data.append(obj)

    def _dump_to_conll(self):
        with open(self.save_path, 'w', encoding='utf8') as fout:
            for example in self.data:
                for i in range(len(example['labels'])):
                    print(
                        example['tokens'][i],
                        example['labels'][i],
                        example['predicts'][i],
                        sep='\t',
                        file=fout,
                    )
                print('', file=fout)

    def _dump_to_jsonline(self):
        with open(self.save_path, 'w', encoding='utf8') as file:
            for example in self.data:
                file.write(json.dumps(example, ensure_ascii=False) + '\n')
