# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys

from adaseq.metainfo import Metrics


class SetScore:
    """evaluate macro and micro set p/r/f1 scores"""

    def __init__(self):
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def reset(self):  # noqa: D102
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def update(self, batch_gold_entities, batch_pred_entities):  # noqa: D102
        self.n_sample += len(batch_gold_entities)
        self.pred.extend(batch_pred_entities)
        self.true.extend(batch_gold_entities)

    def f1(self, precision, recall):  # noqa: D102
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return f1

    def result(self):  # noqa: D102
        assert len(self.pred) == len(self.true)
        M = len(self.pred)
        strict_acc = 0
        num_pred_labels = 0
        num_true_labels = 0
        num_correct_labels = 0
        total_ma_p = 0
        total_ma_r = 0
        total_ma_f1 = 0
        count = 0
        for i in range(M):
            p = set(self.pred[i])
            t = set(self.true[i])
            count += 1

            if p == t:
                strict_acc += 1

            l_p, l_t, l_intersect = len(p), len(t), len(p.intersection(t))
            num_pred_labels += l_p
            num_true_labels += l_t
            num_correct_labels += l_intersect

            if l_p == 0 or l_t == 0:
                ma_p = 0
                ma_r = 0
                ma_f1 = 0
            else:
                ma_p = l_intersect / l_p
                ma_r = l_intersect / l_t
                ma_f1 = self.f1(ma_p, ma_r)

            total_ma_p += ma_p
            total_ma_r += ma_r
            total_ma_f1 += ma_f1

        if num_pred_labels == 0 or num_true_labels == 0:
            micro_p = 0
            micro_r = 0
            micro_f1 = 0
        else:
            micro_p = num_correct_labels / num_pred_labels
            micro_r = num_correct_labels / num_true_labels
            micro_f1 = self.f1(micro_p, micro_r)

        strict_acc /= count
        macro_p = total_ma_p / count
        macro_r = total_ma_r / count
        macro_f1 = self.f1(macro_p, macro_r)
        avg_true_label = num_true_labels / M
        avg_pred_label = num_pred_labels / M

        return {
            'strict_acc': strict_acc,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'avg_true_label': avg_true_label,
            'avg_pred_label': avg_pred_label,
        }


@METRICS.register_module(module_name=Metrics.typing_metric)
class TypingMetric(Metric):
    """Evaluate metrics for typing tasks"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = SetScore()

    def add(self, outputs: Dict, inputs: Dict):  # noqa: D102
        predicts = outputs['predicts']
        pred_results = list()
        ground_truths = list()
        for i, meta in enumerate(inputs['meta']):
            for j, s in enumerate(meta['spans']):
                pred_results.append(predicts[i][j])
                ground_truths.append(set(s['type']))
        self.scorer.update(ground_truths, pred_results)

    def evaluate(self):  # noqa: D102
        score_detail = self.scorer.result()
        scores = {}
        scores[MetricKeys.PRECISION] = score_detail['macro_p']
        scores[MetricKeys.RECALL] = score_detail['macro_r']
        scores[MetricKeys.F1] = score_detail['macro_f1']
        scores[MetricKeys.ACCURACY] = score_detail['strict_acc']
        scores['micro_f1'] = score_detail['micro_f1']
        scores['avg_true_label'] = score_detail['avg_true_label']
        scores['avg_pred_label'] = score_detail['avg_pred_label']

        return scores
