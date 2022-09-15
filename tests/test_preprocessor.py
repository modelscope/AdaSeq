import unittest

from modelscope.preprocessors.builder import build_preprocessor

from uner.preprocessors import *  # noqa


class PreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.input1 = {
            'tokens':
            'EU rejects German call to boycott British lamb .'.split(),
            'labels':
            ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
        }
        self.labels1 = [
            'O', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC',
            'I-ORG', 'I-PER'
        ]
        self.label2id1 = dict(zip(self.labels1, range(len(self.labels1))))

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
        pass


if __name__ == '__main__':
    unittest.main()
