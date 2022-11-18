import os
import unittest

import numpy as np
import torch

from adaseq.modules.decoders import CRF


class TestCRF(unittest.TestCase):
    def setUp(self):
        self.crf = CRF(
            num_tags=2,
            batch_first=True,
        )
        self.crf.start_transitions.data = torch.log(
            torch.ones(self.crf.num_tags) / self.crf.num_tags
        )
        self.crf.end_transitions.data = torch.log(torch.ones(self.crf.num_tags) / self.crf.num_tags)
        self.crf.transitions.data = torch.log(
            torch.tensor(
                [
                    [
                        0.5,
                        0.5,
                    ],
                    [0.5, 0.5],
                ]
            )
        )

        self.case = {
            'feats': torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [1, 1]]])),
        }

    def test_forward_partition(self):
        partition = self.crf._compute_normalizer(
            self.case['feats'].transpose(0, 1),
            torch.tensor([[True, True, False]], dtype=bool).transpose(0, 1),
        )
        self.assertEqual(partition, torch.log(torch.tensor([1 / 8])))

    def test_posterior(self):
        posterior = self.crf.compute_posterior(
            self.case['feats'], torch.tensor([[True, True, False]], dtype=bool)
        )
        np.testing.assert_allclose(
            posterior.detach().numpy()[:, :2, :2], np.log(np.array([[[0.5, 0.5], [0.5, 0.5]]]))
        )


if __name__ == '__main__':
    unittest.main()
