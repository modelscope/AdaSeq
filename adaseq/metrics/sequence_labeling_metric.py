# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify
from seqeval.metrics import accuracy_score, classification_report

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import Metrics


@METRICS.register_module(module_name=Metrics.sequence_labeling_metric)
@METRICS.register_module(module_name=Metrics.ner_metric)
class SequenceLabelingMetric(Metric):
    """The metric computation class for sequence-labeling tasks.

    This metric class uses seqeval to calculate scores.

    Args:
        return_macro_f1 (bool, *optional*):
            Whether to return macro-f1, default False.
        return_class_level_metric (bool, *optional*):
            Whether to return every class's detailed metrics, default False.
    """

    def __init__(
        self, return_macro_f1=False, return_class_level_metric=False, mode=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.return_macro_f1 = return_macro_f1
        self.return_class_level_metric = return_class_level_metric
        self.mode = mode
        self.preds = []
        self.golds = []

    def add(self, outputs: Dict, inputs: Dict):
        """Collect batch outputs"""
        pred_results = outputs['predicts']
        ground_truths = inputs['label_ids']
        self.preds.extend(torch_nested_numpify(torch_nested_detach(pred_results)).tolist())
        self.golds.extend(torch_nested_numpify(torch_nested_detach(ground_truths)).tolist())

    def evaluate(self):
        """Calculate metrics, returning precision, recall, f1-score, accuracy in a dictionary

        Returns:
            scores (Dict):
                precision (float): micro averaged precision
                recall (float): micro averaged recall
                f1 (float): micro averaged f1-score
                accuracy (float): micro averaged accuracy
                tp (Dict): metrics for each label
                    precision (float): precision of each label
                    recall (float): recall of each label
                    f1 (float): f1 of each label
                    support (int): the number of occurrences of each label in ground truth
        """
        id2label = self.trainer.train_preprocessor.id_to_label

        pred_labels = [
            [id2label[p] for p, g in zip(pred, gold) if g != PAD_LABEL_ID]
            for pred, gold in zip(self.preds, self.golds)
        ]
        gold_labels = [[id2label[g] for g in gold if g != PAD_LABEL_ID] for gold in self.golds]

        report = classification_report(gold_labels, pred_labels, output_dict=True, mode=self.mode)

        report.pop('weighted avg')
        macro_score = report.pop('macro avg')
        micro_score = report.pop('micro avg')

        scores = {}
        scores[MetricKeys.PRECISION] = micro_score['precision']
        scores[MetricKeys.RECALL] = micro_score['recall']
        scores[MetricKeys.F1] = micro_score['f1-score']
        scores[MetricKeys.ACCURACY] = accuracy_score(y_true=gold_labels, y_pred=pred_labels)
        if self.return_macro_f1:
            scores['macro-f1'] = macro_score['f1-score']
        if self.return_class_level_metric:
            for tp, tp_score in report.items():
                scores[tp] = {}
                scores[tp][MetricKeys.PRECISION] = round(tp_score['precision'], 4)
                scores[tp][MetricKeys.RECALL] = round(tp_score['recall'], 4)
                scores[tp][MetricKeys.F1] = round(tp_score['f1-score'], 4)
                scores[tp]['support'] = tp_score['support']
        return scores

    def merge(self, other):
        """Merge metrics from multi nodes"""
        self.preds.extend(other.preds)
        self.golds.extend(other.preds)

    def __getstate__(self):
        return (
            self.return_macro_f1,
            self.return_class_level_metric,
            self.mode,
            self.preds,
            self.golds,
        )

    def __setstate__(self, state):
        self.__init__()
        (
            self.return_macro_f1,
            self.return_class_level_metric,
            self.mode,
            self.preds,
            self.golds,
        ) = state
