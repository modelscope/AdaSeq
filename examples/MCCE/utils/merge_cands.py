import os
import pickle
from copy import deepcopy

import inflection
import nltk
import numpy as np
import torch
from modelscope.msdatasets.task_datasets.torch_base_dataset import TorchTaskDataset
from modelscope.utils.checkpoint import load_checkpoint
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys
from modelscope.utils.data_utils import to_device
from modelscope.utils.device import create_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.data.preprocessors.nlp_preprocessor import build_preprocessor
from adaseq.metrics.typing_metric import TypingMetric
from adaseq.models.base import Model

# nltk.download('averaged_perceptron_tagger')
if __name__ == '__main__':
    work_dir = (
        ''  # MLC model path: for example, ADASEQ_ROOT/experiments/ufet-12-30/221231063017.184002'
    )
    em_cands_path = ''
    recall_cands_path = ''
    cands_save_path = ''

    config = Config.from_file(os.path.join(work_dir, 'config.yaml'))
    em_cands = pickle.load(open(em_cands_path, 'rb'))
    recall_cands = pickle.load(open(recall_cands_path, 'rb'))

    dm = DatasetManager.from_config(task=config.task, **config.dataset)
    preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)
    labels = np.array(dm.labels)
    collator_config = config.data_collator
    if isinstance(collator_config, str):
        collator_config = dict(type=collator_config)
    data_collator = build_data_collator(preprocessor.tokenizer, collator_config)

    print('merge two candidates, exact match goes first')
    evaluate_top_k = 512
    metric = TypingMetric()
    cands = dict()
    for dataset_name, dataset in zip(['train', 'dev', 'test'], [dm.train, dm.dev, dm.test]):
        dataset = TorchTaskDataset(dataset, mode=ModeKeys.INFERENCE, preprocessor=preprocessor)
        all_cands = []
        for id, data in tqdm(enumerate(dataset)):
            cand_i = recall_cands[dataset_name][id][:evaluate_top_k]
            cand_i_em = em_cands[dataset_name][id]
            set_cand_i_em = set(cand_i_em)
            cand_i = (
                cand_i_em
                + [i for i in cand_i if i not in set_cand_i_em][: evaluate_top_k - len(cand_i_em)]
            )
            outputs = {'predicts': [[cand_i]]}
            data['meta'] = [data['meta']]
            metric.add(outputs, data)
            all_cands.append(cand_i)
        print(metric.evaluate())

        cands[dataset_name] = deepcopy(all_cands)

    pickle.dump(cands, open(cands_save_path, 'wb'))
