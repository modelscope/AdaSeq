import unittest

from modelscope.preprocessors.builder import build_preprocessor

from uner.preprocessors import *  # noqa


class PreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.input1 = {
            'tokens':
            'EU rejects German call to boycott British lamb .'.split(),
            'spans': [
                {
                    'start': 0,
                    'end': 1,
                    'type': 'ORG'
                },
                {
                    'start': 2,
                    'end': 3,
                    'type': 'MISC'
                },
                {
                    'start': 6,
                    'end': 7,
                    'type': 'MISC'
                },
            ]
        }
        self.labels1 = [
            'O', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC',
            'I-ORG', 'I-PER'
        ]
        self.label2id1 = dict(zip(self.labels1, range(len(self.labels1))))

        self.input2 = {
            'tokens': list('国正先生在我心中就是这样的一位学长。'),
            'spans': [{
                'start': 0,
                'end': 2,
                'type': 'PER'
            }]
        }
        self.labels2 = [
            'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'
        ]
        self.label2id2 = dict(zip(self.labels2, range(len(self.labels2))))

    def test_bert_sequence_labeling_preprocessor(self):
        cfg = dict(
            type='sequence-labeling-preprocessor',
            model_dir='bert-base-cased',
            label2id=self.label2id1)
        preprocessor = build_preprocessor(cfg)
        output1 = preprocessor(self.input1)
        self.assertEqual(output1['input_ids'], [
            101, 7270, 22961, 1528, 1840, 1106, 21423, 1418, 2495, 12913, 119,
            102
        ])
        self.assertEqual(output1['attention_mask'],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(output1['emission_mask'],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])
        self.assertEqual(output1['offset_mapping'], [(0, 0), (0, 1), (1, 2),
                                                     (2, 3), (3, 4), (4, 5),
                                                     (5, 6), (6, 7), (7, 8),
                                                     (8, 8), (8, 9), (0, 0)])
        self.assertEqual(output1['label_ids'], [3, 0, 2, 0, 0, 0, 2, 0, 0])

    def test_lstm_sequence_labeling_preprocessor(self):
        cfg = dict(
            type='sequence-labeling-preprocessor',
            model_dir='pangda/word2vec-skip-gram-mixed-large-chinese',
            label2id=self.label2id2,
            add_cls_sep=False)
        preprocessor = build_preprocessor(cfg)
        output2 = preprocessor(self.input2)
        self.assertEqual(output2['input_ids'], [
            215, 209, 284, 264, 6, 22, 389, 27, 39, 9, 40, 1389, 3, 14, 138,
            350, 177, 4
        ])
        self.assertEqual(
            output2['attention_mask'],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(
            output2['emission_mask'],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(output2['offset_mapping'], [(0, 1), (1, 2), (2, 3),
                                                     (3, 4), (4, 5), (5, 6),
                                                     (6, 7), (7, 8), (8, 9),
                                                     (9, 10), (10, 11),
                                                     (11, 12), (12, 13),
                                                     (13, 14), (14, 15),
                                                     (15, 16), (16, 17),
                                                     (17, 18)])
        self.assertEqual(
            output2['label_ids'],
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
