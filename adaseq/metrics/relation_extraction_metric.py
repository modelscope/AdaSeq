# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify

from adaseq.data.constant import NONE_REL_LABEL
from adaseq.metainfo import Metrics


def compute_f1(preds: List[str], labels: List[str]) -> Dict[str, float]:
    """Compute f1 from predictions and ground truth

    Args:
        preds (List[str]): prediction list
        labels (List[str]): ground truth list

    Returns:
        A dict containing: precision, recall, f1-score
    """
    # From https://github.com/princeton-nlp/PURE/blob/main/run_relation.py
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != NONE_REL_LABEL:
            n_pred += 1
        if label != NONE_REL_LABEL:
            n_gold += 1
        if (pred != NONE_REL_LABEL) and (label != NONE_REL_LABEL) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        return {'precision': prec, 'recall': recall, 'f1': f1}


@METRICS.register_module(module_name=Metrics.relation_extraction_metric)
class RelationExtractionMetric(Metric):
    """The metric computation class for relation extraction tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.golds = []

    def add(self, outputs: Dict, inputs: Dict):
        """Collect batch outputs"""
        pred_results = outputs['predicts']
        ground_truths = inputs['label_id']
        self.preds.extend(torch_nested_numpify(torch_nested_detach(pred_results)).tolist())
        self.golds.extend(torch_nested_numpify(torch_nested_detach(ground_truths)).tolist())

    def evaluate(self):
        """Calculate metrics, returning precision, recall, f1-score in a dictionary

        Returns:
            scores (Dict):
                precision (float): micro averaged precision
                recall (float): micro averaged recall
                f1 (float): micro averaged f1-score
        """
        id2label = self.trainer.train_preprocessor.id_to_label

        pred_labels = [id2label[p] for p in self.preds]
        gold_labels = [id2label[p] for p in self.golds]

        overall_score = compute_f1(pred_labels, gold_labels)

        scores = {}
        scores[MetricKeys.PRECISION] = overall_score['precision']
        scores[MetricKeys.RECALL] = overall_score['recall']
        scores[MetricKeys.F1] = overall_score['f1']
        return scores

    def merge(self, other):
        """Merge metrics from multi nodes"""
        self.preds.extend(other.preds)
        self.golds.extend(other.preds)

    def __getstate__(self):
        return (self.preds, self.golds)

    def __setstate__(self, state):
        self.__init__()
        (self.preds, self.golds) = state
