# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import defaultdict
from typing import Dict, List, Set

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS

from adaseq.data.span_utils import TypedSpan
from adaseq.metainfo import Metrics

UNLABELED_KEY = 'UNLABELED'
MACRO_KEY = 'MACRO'


@METRICS.register_module(module_name=Metrics.span_extraction_metric)
class SpanExtractionMetric(Metric):
    """The metric computation class for span-extraction tasks."""

    def __init__(self, *args, **kwargs):
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def _update(self, predicted_spans: List[TypedSpan], gold_spans: List[TypedSpan]):
        """
        Update counters.
        """
        gold = set(gold_spans)
        for span in predicted_spans:
            if span in gold:
                self._true_positives[span[2]] += 1
                gold.remove(span)
            else:
                self._false_positives[span[2]] += 1
        # These spans weren't predicted.
        for span in gold:
            self._false_negatives[span[2]] += 1

    def add(self, outputs: Dict, inputs: Dict):
        """Update counters."""
        for predicted_spans, meta in zip(outputs['predicts'], inputs['meta']):
            gold_spans = [(s['start'], s['end'], s['type']) for s in meta['spans']]
            self._update(predicted_spans, gold_spans)
            predicted_unlabeled = [(s[0], s[1], UNLABELED_KEY) for s in predicted_spans]
            gold_unlabeled = [(s[0], s[1], UNLABELED_KEY) for s in gold_spans]
            self._update(predicted_unlabeled, gold_unlabeled)

    def evaluate(self) -> Dict[str, float]:
        """
        Returns:
        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1 : `float`
            Additionally, an `UNLABELED` key for unlabeled precision, recall and f1,
            a `MACRO` key for macro averaged scores.
        """
        # gather all types, since come counter may have missing keys.
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        # compute per class metrics, including unlabeled scores.
        all_metrics = dict()
        for tag in all_tags:
            all_metrics[tag] = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )

        # pop unlabeled scores to aggregate macro scores
        unlabeled_metrics = all_metrics.pop(UNLABELED_KEY)
        # macro scores
        macro_p = sum(v['precision'] for v in all_metrics.values()) / len(all_metrics)
        macro_r = sum(v['recall'] for v in all_metrics.values()) / len(all_metrics)
        macro_f1 = 2.0 * (macro_p * macro_r) / (macro_p + macro_r + 1e-13)
        all_metrics[MACRO_KEY] = dict(precision=macro_p, recall=macro_r, f1=macro_f1)
        # put unlabeled scores back
        all_metrics[UNLABELED_KEY] = unlabeled_metrics

        for v in (self._true_positives, self._false_positives, self._false_negatives):
            v.pop(UNLABELED_KEY)
        # overall micro scores
        micro_metrics = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )

        scores = micro_metrics
        all_tags.remove(UNLABELED_KEY)
        all_types = [MACRO_KEY, UNLABELED_KEY] + sorted(all_tags)
        for name in all_types:
            for k, v in all_metrics[name].items():
                scores[f'{k}-{name}'] = v
        return scores

    @staticmethod
    def _compute_metrics(
        true_positives: int, false_positives: int, false_negatives: int
    ) -> Dict[str, float]:
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return dict(precision=precision, recall=recall, f1=f1)

    def merge(self, other):  # noqa: D102
        raise NotImplementedError
