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

        self.case_1 = {
            'feats': torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [1.0, 1.0]]])),
            'masks': torch.tensor([[True, True, False]], dtype=bool),
            'transition': torch.log(
                torch.tensor(
                    [
                        [
                            0.5,
                            0.5,
                        ],
                        [0.5, 0.5],
                    ]
                )
            ),
            'partition': torch.log(torch.tensor([0.125])),
            'posterior': torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])),
        }

        self.case_2 = {
            'feats': torch.log(torch.tensor([[[0.1, 0.9], [0.3, 0.7], [1.0, 1.0]]])),
            'masks': torch.tensor([[True, True, False]], dtype=bool),
            'transition': torch.log(
                torch.tensor(
                    [
                        [
                            0.5,
                            0.5,
                        ],
                        [0.5, 0.5],
                    ]
                )
            ),
            'partition': torch.log(torch.tensor([0.125])),
            'posterior': torch.log(torch.tensor([[[0.1, 0.9], [0.3, 0.7]]])),
        }

        self.case_3 = {
            'feats': torch.log(torch.tensor([[[0.1, 0.9], [0.3, 0.7], [1.0, 1.0]]])),
            'masks': torch.tensor([[True, True, False]], dtype=bool),
            'transition': torch.log(
                torch.tensor(
                    [
                        [
                            0.1,
                            0.9,
                        ],
                        [0.4, 0.6],
                    ]
                )
            ),
            'partition': torch.log(torch.tensor([0.138])),
            'posterior': torch.log(
                torch.tensor(
                    [[[(3 + 63) / 552, (108 + 378) / 552], [(3 + 108) / 552, (63 + 378) / 552]]]
                )
            ),
        }

        self.cases = [self.case_1, self.case_2, self.case_3]

    def test_forward_partition(self):
        for i, case in enumerate(self.cases):
            self.crf.transitions.data = case['transition']
            partition = self.crf._compute_normalizer(
                case['feats'].transpose(0, 1),
                case['masks'].transpose(0, 1),
            )
            np.testing.assert_allclose(
                partition.detach().numpy(), case['partition'].detach().numpy()
            )

    def test_posterior(self):
        for i, case in enumerate(self.cases):
            self.crf.transitions.data = case['transition']
            posterior = self.crf.compute_posterior(case['feats'], case['masks'])
            np.testing.assert_allclose(
                posterior.detach().numpy()[:, :2, :2], case['posterior'].detach().numpy(), rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
