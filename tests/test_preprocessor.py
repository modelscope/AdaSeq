import unittest

import numpy as np
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.utils.constant import ModeKeys
from transformers import BertTokenizerFast

from adaseq.data.preprocessors import NLPPreprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.setUp_ner_examples()
        self.setUp_typing_examples()

    def setUp_ner_examples(self):
        self.ner_input1 = {
            'tokens': 'EU rejects German call to boycott British lamb .'.split(),
            'spans': [
                {'start': 0, 'end': 1, 'type': 'ORG'},
                {'start': 2, 'end': 3, 'type': 'MISC'},
                {'start': 6, 'end': 7, 'type': 'MISC'},
            ],
        }
        self.ner_labels = ['LOC', 'MISC', 'ORG', 'PER']

        self.ner_input2 = {
            'tokens': list('国正先生在我心中就是这样的一位学长。'),
            'spans': [{'start': 0, 'end': 2, 'type': 'PER'}],
        }

        self.ner_labels2 = ['PER', 'ORG', 'LOC']

        self.ner_input3 = {'text': 'EU rejects German call to boycott British lamb .'}

    def test_bert_sequence_labeling_preprocessor(self):
        cfg = dict(
            type='sequence-labeling-preprocessor',
            model_dir='bert-base-cased',
            labels=self.ner_labels,
            tag_scheme='BIO',
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        id_to_label = {v: k for k, v in preprocessor.label_to_id.items()}
        labels = ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
        output1 = preprocessor(self.ner_input1)
        input_ids = [101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119, 102]
        self.assertEqual(output1['tokens']['input_ids'], input_ids)
        self.assertEqual(output1['tokens']['attention_mask'], [True] * 12)
        # add_special_tokens=True by default
        self.assertEqual(output1['tokens']['mask'], [True] * (len(labels) + 2))
        offsets = [(i, i) for i in range(8)] + [(8, 9), (10, 10), (11, 11)]
        self.assertEqual(output1['tokens']['offsets'], offsets)
        output_labels = [id_to_label[i] for i in output1['label_ids']]
        self.assertEqual(output_labels, labels)

    def test_bert_sequence_labeling_bioes_preprocessor(self):
        cfg = dict(
            type='sequence-labeling-preprocessor',
            model_dir='bert-base-cased',
            labels=self.ner_labels,
            tag_scheme='BIOES',
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        id_to_label = {v: k for k, v in preprocessor.label_to_id.items()}
        output1 = preprocessor(self.ner_input1)
        labels = ['S-ORG', 'O', 'S-MISC', 'O', 'O', 'O', 'S-MISC', 'O', 'O']
        output_labels = [id_to_label[i] for i in output1['label_ids']]
        self.assertEqual(output_labels, labels)

    def test_word2vec_sequence_labeling_preprocessor(self):
        cfg = dict(
            type='sequence-labeling-preprocessor',
            model_dir='pangda/word2vec-skip-gram-mixed-large-chinese',
            labels=self.ner_labels,
            add_special_tokens=False,
            tag_scheme='BIOES',
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        id_to_label = {v: k for k, v in preprocessor.label_to_id.items()}
        output2 = preprocessor(self.ner_input2)
        input_ids = [217, 211, 286, 266, 8, 24, 391, 29, 41, 11, 42, 1391, 5, 16, 140, 352, 179, 6]
        labels = [
            'B-PER',
            'E-PER',
        ] + ['O'] * 16
        self.assertEqual(output2['tokens']['input_ids'], input_ids)
        self.assertEqual(output2['tokens']['mask'], [True] * len(labels))
        output_labels = [id_to_label[i] for i in output2['label_ids']]
        self.assertEqual(output_labels, labels)

    def test_bert_span_extraction_preprocessor(self):
        cfg = dict(
            type='span-extraction-preprocessor',
            model_dir='bert-base-cased',
            labels=self.ner_labels2,
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        output2 = preprocessor(self.ner_input2)
        expected_labels = np.zeros((18, 18))
        expected_labels[0][1] = preprocessor.label_to_id['PER'] + 1
        self.assertEqual((output2['span_labels'] == expected_labels).all(), True)

    def setUp_typing_examples(self):
        self.typing_input = {
            'tokens': list('国正先生在我心中就是这样的一位学长。'),
            'spans': [{'start': 0, 'end': 2, 'type': ['人名', '老师']}],
        }

        self.typing_labels = ['人名', '地名', '老师']

    def test_span_typing_preprocessor(self):
        cfg = dict(
            type='multilabel-span-typing-preprocessor',
            model_dir='bert-base-cased',
            labels=self.typing_labels,
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        id_to_label = {v: k for k, v in preprocessor.label_to_id.items()}
        output = preprocessor(self.typing_input)
        input_ids = [
            101,
            1004,
            1045,
            100,
            1056,
            100,
            100,
            1027,
            980,
            100,
            100,
            100,
            100,
            100,
            976,
            100,
            100,
            100,
            886,
            102,
        ]
        self.assertEqual(output['tokens']['input_ids'], input_ids)
        self.assertEqual(output['tokens']['attention_mask'], [True] * len(input_ids))
        self.assertEqual(output['tokens']['mask'], [True] * len(input_ids))
        self.assertEqual(output['tokens']['offsets'], [(i, i) for i in range(len(input_ids))])
        self.assertEqual(output['mention_boundary'], [[0], [1]])
        output_labels = [id_to_label[i] for i in output['type_ids'][0]]
        self.assertEqual([output_labels], [['地名', '人名', '地名']])

    def test_load_modelscope_tokenizer(self):
        processor = NLPPreprocessor(
            model_dir='damo/nlp_structbert_backbone_tiny_std', labels=['O', 'B']
        )
        self.assertTrue(isinstance(processor.tokenizer, BertTokenizerFast))

    def test_text_preprocessor_train(self):
        cfg = dict(
            type='nlp-preprocessor',
            model_dir='bert-base-cased',
            return_offsets=True,
            labels=['O'],
            mode=ModeKeys.TRAIN,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        output = preprocessor(self.ner_input3)
        input_ids = [
            101,
            7270,
            22961,
            1528,
            1840,
            1106,
            21423,
            1418,
            2495,
            108,
            108,
            182,
            1830,
            119,
            102,
        ]
        offsets = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 12),
            (13, 13),
            (14, 14),
        ]
        self.assertTrue(np.array_equal(output['tokens']['input_ids'], input_ids))
        self.assertTrue(np.array_equal(output['tokens']['attention_mask'], [True] * 15))
        self.assertTrue(np.array_equal(output['tokens']['offsets'], offsets))

    def test_text_preprocessor_inference(self):
        cfg = dict(
            type='nlp-preprocessor',
            model_dir='bert-base-cased',
            labels=['O'],
            mode=ModeKeys.INFERENCE,
        )
        preprocessor = build_preprocessor(cfg, 'nlp')
        output = preprocessor(self.ner_input3)
        self.assertEqual(output['tokens']['input_ids'].ndim, 2)
        self.assertEqual(output['tokens']['input_ids'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
