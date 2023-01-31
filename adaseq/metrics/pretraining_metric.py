# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify
from seqeval.metrics import accuracy_score, classification_report

from adaseq.data.constant import PAD_LABEL_ID
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


@METRICS.register_module(module_name=Metrics.pretraining_metric)
class PretrainingMetric(Metric):
    """Evaluate metrics for typing tasks and identification tasks"""

    def __init__(self, return_class_level_metric=False, mode=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.typing_cls_scorer = SetScore()
        self.return_class_level_metric = return_class_level_metric
        self.mode = mode
        self.ident_preds = []
        self.ident_golds = []
        self.prompt_preds = []
        self.prompt_golds = []

    def add(self, outputs: Dict, inputs: Dict):  # noqa: D102
        ident_pred_results, typing_predicts, prompt_pred_results = outputs['predicts']
        typing_pred_results = list()
        typing_ground_truths = list()
        for i, meta in enumerate(inputs['meta']):
            for j, s in enumerate(meta['spans']):
                typing_pred_results.append(typing_predicts[i][j])
                typing_ground_truths.append(set(s['type']))
        self.typing_cls_scorer.update(typing_ground_truths, typing_pred_results)

        ident_ground_truths = inputs['ident_ids']
        self.ident_preds.extend(
            torch_nested_numpify(torch_nested_detach(ident_pred_results)).tolist()
        )
        self.ident_golds.extend(
            torch_nested_numpify(torch_nested_detach(ident_ground_truths)).tolist()
        )
        prompt_ground_truths = inputs['prompt_target_label_ids']
        self.prompt_preds.extend(
            torch_nested_numpify(torch_nested_detach(prompt_pred_results)).tolist()
        )
        self.prompt_golds.extend(
            torch_nested_numpify(torch_nested_detach(prompt_ground_truths)).tolist()
        )

    def evaluate(self):  # noqa: D102

        # for detection
        id2label = self.trainer.train_preprocessor.ident_id_to_label

        pred_labels = [
            [id2label[p] for p, g in zip(ident_pred, ident_gold) if g != PAD_LABEL_ID]
            for ident_pred, ident_gold in zip(self.ident_preds, self.ident_golds)
        ]
        gold_labels = [
            [id2label[g] for g in ident_gold if g != PAD_LABEL_ID]
            for ident_gold in self.ident_golds
        ]

        ident_report = classification_report(
            gold_labels, pred_labels, output_dict=True, mode=self.mode
        )

        ident_report.pop('macro avg')
        ident_report.pop('weighted avg')
        ident_overall_score = ident_report.pop('micro avg')

        scores = {}
        scores['ident_precision'] = ident_overall_score['precision']
        scores['ident_recall'] = ident_overall_score['recall']
        scores['ident_F1'] = ident_overall_score['f1-score']
        scores['ident_accuracy'] = accuracy_score(y_true=gold_labels, y_pred=pred_labels)
        if self.return_class_level_metric:
            for tp, tp_score in ident_report.items():
                scores[tp] = {}
                scores[tp][MetricKeys.PRECISION] = round(tp_score['precision'], 4)
                scores[tp][MetricKeys.RECALL] = round(tp_score['recall'], 4)
                scores[tp][MetricKeys.F1] = round(tp_score['f1-score'], 4)
                scores[tp]['support'] = tp_score['support']
        # for typing
        typing_score_detail = self.typing_cls_scorer.result()

        scores['typing_macro_f1'] = typing_score_detail['macro_f1']
        scores['typing_macro_precision'] = typing_score_detail['macro_p']
        scores['typing_macro_recall'] = typing_score_detail['macro_r']

        scores['typing_micro_f1'] = typing_score_detail['micro_f1']
        scores['typing_micro_precision'] = typing_score_detail['micro_p']
        scores['typing_micro_recall'] = typing_score_detail['micro_r']

        scores['typing_strict_acc'] = typing_score_detail['strict_acc']
        scores['avg_true_label of typing'] = typing_score_detail['avg_true_label']
        scores['avg_pred_label of typing'] = typing_score_detail['avg_pred_label']

        scores[MetricKeys.PRECISION] = typing_score_detail['macro_p']
        scores[MetricKeys.RECALL] = typing_score_detail['macro_r']
        scores[MetricKeys.F1] = typing_score_detail['macro_f1']
        scores[MetricKeys.ACCURACY] = typing_score_detail['strict_acc']
        # prompt ner
        pred_labels = [
            [id2label[p] for p, g in zip(prompt_pred, prompt_gold) if g != PAD_LABEL_ID]
            for prompt_pred, prompt_gold in zip(self.prompt_preds, self.prompt_golds)
        ]
        gold_labels = [
            [id2label[g] for g in prompt_gold if g != PAD_LABEL_ID]
            for prompt_gold in self.prompt_golds
        ]
        prompt_report = classification_report(
            gold_labels, pred_labels, output_dict=True, mode=self.mode
        )

        prompt_report.pop('macro avg')
        prompt_report.pop('weighted avg')
        prompt_overall_score = prompt_report.pop('micro avg')

        scores['prompt_precision'] = prompt_overall_score['precision']
        scores['prompt_recall'] = prompt_overall_score['recall']
        scores['prompt_F1'] = prompt_overall_score['f1-score']
        scores['prompt_accuracy'] = accuracy_score(y_true=gold_labels, y_pred=pred_labels)

        return scores

    def merge(self, other):
        """Merge metrics from multi nodes"""
        self.ident_preds.extend(other.ident_preds)
        self.ident_golds.extend(other.ident_golds)
        self.prompt_preds.extend(other.prompt_preds)
        self.prompt_golds.extend(other.prompt_golds)

    def __getstate__(self):
        return (
            self.typing_cls_scorer,
            self.return_class_level_metric,
            self.mode,
            self.ident_preds,
            self.ident_golds,
            self.prompt_preds,
            self.prompt_golds,
        )

    def __setstate__(self, state):
        self.__init__()
        (
            self.typing_cls_scorer,
            self.return_class_level_metric,
            self.mode,
            self.ident_preds,
            self.ident_golds,
            self.prompt_preds,
            self.prompt_golds,
        ) = state
