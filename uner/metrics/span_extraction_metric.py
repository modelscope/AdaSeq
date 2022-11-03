from collections import Counter
from typing import Dict

from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.tensor_utils import (
    torch_nested_detach,
    torch_nested_numpify,
)
from seqeval.metrics import accuracy_score, classification_report

from uner.metainfo import Metrics


class EntityScore:

    def __init__(self):
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []
        self.loc_rights = []
        self.n_sentences = 0

    def _compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (
            precision + recall)
        return recall, precision, f1

    def result(self):
        detailed_report = 'evaluation report:\n'
        origin_counter = Counter([x['type'] for x in self.origins])
        found_counter = Counter([x['type'] for x in self.founds])
        right_counter = Counter([x['type'] for x in self.rights])
        for type, count in origin_counter.items():
            origin = count
            found = found_counter.get(type, 0)
            right = right_counter.get(type, 0)
            recall, precision, f1 = self._compute(origin, found, right)
            detailed_report += 'Type: %s - precision: %.4f - recall: %.4f - f1: %.4f\n' % (
                type, precision, recall, f1)
        # ALL
        origin = sum(origin_counter.values())
        found = sum(found_counter.values())
        right = sum(right_counter.values())
        recall, precision, f1 = self._compute(origin, found, right)
        detailed_report += 'All Types - precision: %.4f - recall: %.4f - f1: %.4f\n' % (
            precision, recall, f1)
        # loc level
        loc_right = len(self.loc_rights)
        loc_recall, loc_precision, loc_f1 = self._compute(
            origin, found, loc_right)
        detailed_report += 'All Types(loc) - precision: %.4f - recall: %.4f - f1: %.4f\n' % (
            loc_precision, loc_recall, loc_f1)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detailed_report': detailed_report
        }

    @staticmethod
    def join_entities(pre_entities, label_entities):
        ret = []
        for pre_entity in pre_entities:
            for label_entity in label_entities:
                if pre_entity['start'] == label_entity['start'] and pre_entity[
                        'end'] == label_entity['end']:
                    ret.append(pre_entity)
                    break
        return ret

    def update(self, batch_gold_entities, batch_pred_entities):
        self.n_sentences += len(batch_gold_entities)
        for gold_entities, pred_entities in zip(batch_gold_entities,
                                                batch_pred_entities):
            if isinstance(gold_entities[0]['type'], list):
                expanded_golden_mentions = []
                for mention in gold_entities:
                    expanded_golden_mentions.extend([{
                        'start': mention['start'],
                        'end': mention['end'],
                        'type': t
                    } for t in mention['type']])
                gold_entities = expanded_golden_mentions
                expanded_pred_mentions = []
                for mention in pred_entities:
                    expanded_pred_mentions.extend([{
                        'start': mention['start'],
                        'end': mention['end'],
                        'type': t
                    } for t in mention['type']])
                pred_entities = expanded_pred_mentions

            self.origins.extend(gold_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([
                pre_entity for pre_entity in pred_entities
                if pre_entity in gold_entities
            ])
            self.loc_rights.extend(
                self.join_entities(pred_entities, gold_entities))


@METRICS.register_module(module_name=Metrics.span_extraction_metric)
class SpanExtractionMetric(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorer = EntityScore()

    def add(self, outputs: Dict, inputs: Dict):
        token_mapping = inputs['offset_mapping']
        id2label = self.trainer.id2label
        pred_results = outputs['predicts']
        pred_entities_batch = []
        for i, pred_result in enumerate(pred_results):
            pred_entities = []
            for span in pred_result:
                start = token_mapping[i][span[0]][0]
                end = token_mapping[i][span[1]][1]
                if isinstance(span[2], list):
                    typ = [id2label[x] for x in span[2]]
                else:
                    typ = id2label[span[2]]
                pred_entities.append({'start': start, 'end': end, 'type': typ})
            pred_entities_batch.append(pred_entities)

        ground_truths = inputs['spans']
        self.scorer.update(ground_truths, pred_entities_batch)

    def evaluate(self):
        score_detail = self.scorer.result()
        scores = {}
        scores[MetricKeys.PRECISION] = score_detail['precision']
        scores[MetricKeys.RECALL] = score_detail['recall']
        scores[MetricKeys.F1] = score_detail['f1']
        scores[MetricKeys.ACCURACY] = score_detail['precision']
        return scores
