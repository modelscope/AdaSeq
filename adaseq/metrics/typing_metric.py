# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from typing import Dict

import numpy as np
import torch
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys

from adaseq.metainfo import Metrics

logger = logging.getLogger(__name__)


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

    def set_pred_true(self, pred, true):  # noqa: D102
        self.pred = pred
        self.true = true

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
        for i, golden_type_ids in enumerate(inputs['type_ids']):
            for j in range(len(golden_type_ids)):

                def one_hot_to_list(in_tensor):
                    id_list = set((np.where(in_tensor.detach().cpu() == 1)[0]))
                    return id_list

                ground_truths.append(one_hot_to_list(golden_type_ids[j]))
                if j < len(predicts[i]):
                    pred_results.append(one_hot_to_list(predicts[i][j]))
                else:
                    pred_results.append(set([]))
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

    def merge(self, other):  # noqa: D102
        raise NotImplementedError


@METRICS.register_module(module_name=Metrics.typing_threshold_metric)
class ConcatTypingThresholdMetric(Metric):
    """Evaluate metrics for typing tasks, with threshold tuning"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = SetScore()
        self.all_logits = []
        self.all_cands = []
        self.all_true = []

    def add(self, outputs: Dict, inputs: Dict):  # noqa: D102
        batch_logits = outputs['logits'].sigmoid().detach().cpu().numpy()
        B, C = batch_logits.shape[0], batch_logits.shape[-1]
        label_to_id = outputs['label_to_id']  # convert labels to id for easier evaluation

        if 'cands' in inputs:
            batch_cands = inputs['cands'].cpu().numpy()
        else:
            batch_cands = torch.arange(C).repeat(B, 1).cpu().numpy()

        ground_truths = list()
        for i, meta in enumerate(inputs['meta']):
            for j, s in enumerate(meta['spans']):
                ground_truths.append(set([label_to_id[j] for j in s['type']]))

        self.all_true += ground_truths
        self.all_cands.append(batch_cands)
        self.all_logits.append(batch_logits)

    def tune_threshold(self):
        """
        TODO
        """
        th = np.linspace(self.all_logits.min(), self.all_logits.max(), 10)
        all_res = []

        all_res_detail = []
        # stage 1 tuning
        for _th in th:
            pred_ = [
                set(self.all_cands[i][self.all_logits[i] > _th])
                for i in range(len(self.all_logits))
            ]
            self.scorer.set_pred_true(pred_, self.all_true)
            res = self.scorer.result()
            res_val = res['macro_f1'].__round__(5)
            all_res.append(res_val)
            all_res_detail.append(res)

        logger.info('phase 1: all thresholds: {}'.format(th))
        logger.info('phase 1: all results: {}'.format(all_res))

        best = np.argsort(all_res)[-1]
        best_1_th = th[best]
        new_range = (max(best_1_th - 0.15, 0), min(best_1_th + 0.15, 1.0))
        step = (new_range[1] - new_range[0]) / 30
        new_th = np.arange(new_range[0], new_range[1], step)
        all_res = []
        all_res_detail = []

        for _th in new_th:
            pred_ = [
                set(self.all_cands[i][self.all_logits[i] > _th])
                for i in range(len(self.all_logits))
            ]
            self.scorer.set_pred_true(pred_, self.all_true)
            res = self.scorer.result()
            res_val = res['macro_f1'].__round__(5)
            all_res.append(res_val)
            all_res_detail.append(res)

        best_th, best_th_res = new_th[np.argmax(all_res)], all_res_detail[np.argmax(all_res)]
        logger.info('phase 2: all thresholds: {}'.format(new_th))
        logger.info('phase 2: all results: {}'.format(all_res))

        logger.info('phase 2: best threshold: {}'.format(best_th))

        return best_th, best_th_res

    def apply_threshold(self, th):
        """
        TODO
        """
        pred_ = [
            set(self.all_cands[i][self.all_logits[i] > th]) for i in range(len(self.all_logits))
        ]
        self.scorer.set_pred_true(pred_, self.all_true)
        res = self.scorer.result()
        return th, res

    def evaluate(self):  # noqa: D102
        self.all_cands = np.concatenate(self.all_cands)
        self.all_logits = np.concatenate(self.all_logits)

        if getattr(self.trainer, 'do_test', False):
            best_th, score_detail = self.apply_threshold(th=getattr(self.trainer, 'best_th', 0.5))
        else:
            best_th, score_detail = self.tune_threshold()
            self.trainer.best_th = best_th

        scores = {}
        scores[MetricKeys.PRECISION] = score_detail['macro_p']
        scores[MetricKeys.RECALL] = score_detail['macro_r']
        scores[MetricKeys.F1] = score_detail['macro_f1']
        scores[MetricKeys.ACCURACY] = score_detail['strict_acc']
        scores['micro_f1'] = score_detail['micro_f1']
        scores['avg_true_label'] = score_detail['avg_true_label']
        scores['avg_pred_label'] = score_detail['avg_pred_label']

        return scores

    def merge(self, other):  # noqa: D102
        raise NotImplementedError
