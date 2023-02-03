import os
import pickle
from copy import deepcopy

import inflection
import nltk
import numpy as np
from modelscope.msdatasets.task_datasets.torch_base_dataset import TorchTaskDataset
from modelscope.utils.config import Config
from modelscope.utils.constant import ModeKeys
from tqdm import tqdm

from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.data.preprocessors.nlp_preprocessor import build_preprocessor
from adaseq.metrics.typing_metric import TypingMetric

# nltk.download('averaged_perceptron_tagger')
if __name__ == '__main__':
    work_dir = (
        ''  # MLC model path: for example, ADASEQ_ROOT/experiments/ufet-12-30/221231063017.184002'
    )
    cands_save_path = ''  # CAND_save_path: for example: ADASEQ_ROOT/examples/MCCE/cands/em.cand'
    config = Config.from_file(os.path.join(work_dir, 'config.yaml'))
    config.model.top_k = 128
    dm = DatasetManager.from_config(task=config.task, **config.dataset)
    preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)
    labels = np.array(dm.labels)
    collator_config = config.data_collator
    if isinstance(collator_config, str):
        collator_config = dict(type=collator_config)
    data_collator = build_data_collator(preprocessor.tokenizer, collator_config)
    metric = TypingMetric()
    cands = dict()

    original_types = dm.labels
    types = [j.lower().replace('_', ' ') for j in dm.labels]
    types_to_id = {t: id for id, t in enumerate(types)}
    set_types = set(types)

    def get_2_order_phrase(tokens):
        all = []
        for i in range(len(tokens) - 1):
            all.append(tokens[i] + ' ' + tokens[i + 1])
        return all

    for dataset_name, dataset in zip(['train', 'dev', 'test'], [dm.train, dm.dev, dm.test]):
        dataset = TorchTaskDataset(dataset, mode=ModeKeys.INFERENCE, preprocessor=preprocessor)
        all_cands = []
        for data in tqdm(dataset):
            ctx_words = [i.lower() for i in data['meta']['tokens']]
            pos_tags = nltk.pos_tag(ctx_words)
            words = [word for word, pos in pos_tags if pos in ['NNP', 'NN', 'NNS', 'NNPS']]
            singular_words = [inflection.singularize(i) for i in words]

            set_2_order = get_2_order_phrase(words)
            set_2_order_singlular = get_2_order_phrase(singular_words)
            set_words = set(words + singular_words + set_2_order + set_2_order_singlular)
            types = set_words.intersection(set_types)
            cands_i = [original_types[types_to_id[i]] for i in types]
            outputs = {'predicts': [[cands_i]]}
            data['meta'] = [data['meta']]
            metric.add(outputs, data)

            all_cands.append(cands_i)

        print(metric.evaluate())

        cands[dataset_name] = deepcopy(all_cands)

    pickle.dump(cands, open(cands_save_path, 'wb'))
