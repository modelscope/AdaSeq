import os
import pickle
from copy import deepcopy

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

if __name__ == '__main__':
    work_dir = (
        ''  # MLC model path: for example, ADASEQ_ROOT/experiments/ufet-12-30/221231063017.184002'
    )
    cands_save_path = (
        ''  # CAND_save_path: for example: ADASEQ_ROOT/examples/MCCE/cands/npcrf.512.cand'
    )
    checkpoint_name = 'best_model.pth'
    device_name = 'cuda:0'
    device = create_device(device_name=device_name)
    config = Config.from_file(os.path.join(work_dir, 'config.yaml'))
    config.model.top_k = 128
    checkpoint_path = os.path.join(work_dir, checkpoint_name)
    model = Model.from_config(config)
    load_checkpoint(checkpoint_path, model)
    if device.type == 'cuda':
        model.to(device)
    # model = Model.from_pretrained(model_name_or_path=checkpoint_path)
    dm = DatasetManager.from_config(task=config.task, **config.dataset)
    preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)
    labels = np.array(dm.labels)
    collator_config = config.data_collator
    if isinstance(collator_config, str):
        collator_config = dict(type=collator_config)
    data_collator = build_data_collator(preprocessor.tokenizer, collator_config)
    metric = TypingMetric()
    cands = dict()

    for dataset_name, dataset in zip(['train', 'dev', 'test'], [dm.train, dm.dev, dm.test]):
        dataset = TorchTaskDataset(dataset, mode=ModeKeys.INFERENCE, preprocessor=preprocessor)
        data_loader = DataLoader(dataset, batch_size=4, collate_fn=data_collator, shuffle=False)
        all_logits, all_cands = [], []
        with torch.no_grad():
            for i, data in tqdm(enumerate(data_loader)):
                data = to_device(data, device)
                predict = model.forward(**data)
                probs = predict['logits'].sigmoid()
                logits, indices = probs.topk(512)
                all_logits.append(logits.cpu())
                all_cands.append(indices.cpu())
                metric.add(predict, data)

        print(metric.evaluate())
        all_logits = torch.cat(all_logits).numpy()
        all_cands = torch.cat(all_cands).numpy()
        all_cands = labels[all_cands]
        cands[dataset_name] = deepcopy(all_cands)

    pickle.dump(cands, open(cands_save_path, 'wb'))
