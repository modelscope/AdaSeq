from typing import Dict

from seqeval.metrics import accuracy_score, classification_report
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.tensor_utils import torch_nested_detach, torch_nested_numpify

from uner.metainfo import Metrics
from uner.preprocessors.constant import PAD_LABEL_ID


@METRICS.register_module(module_name=Metrics.ner_metric)
class NERMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.golds = []

    def add(self, outputs: Dict, inputs: Dict):
        pred_results = outputs['predicts']
        ground_truths = inputs['label_ids']
        self.preds.extend(
            torch_nested_numpify(torch_nested_detach(pred_results)).tolist())
        self.golds.extend(
            torch_nested_numpify(torch_nested_detach(ground_truths)).tolist())

    def evaluate(self):
        id2label = self.trainer.id2label
        
        pred_labels = [[
            id2label[p] for p, g in zip(pred, gold) if g != PAD_LABEL_ID
        ] for pred, gold in zip(self.preds, self.golds)]
        gold_labels = [[
            id2label[g] for g in gold if g != PAD_LABEL_ID
        ] for gold in self.golds]
            
        report = classification_report(
            gold_labels,
            pred_labels,
            output_dict=True
        )

        report.pop('macro avg')
        report.pop('weighted avg')
        overall_score = report.pop('micro avg')

        scores = {}
        scores[MetricKeys.PRECISION] = overall_score['precision']
        scores[MetricKeys.RECALL] = overall_score['recall']
        scores[MetricKeys.F1] = overall_score['f1-score']
        scores[MetricKeys.ACCURACY] = accuracy_score(y_true=gold_labels, y_pred=pred_labels)
        return scores

