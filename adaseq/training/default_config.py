# Copyright (c) Alibaba, Inc. and its affiliates.
DEFAULT_CONFIG = {
    'train': {
        'dataloader': {'workers_per_gpu': 0},
        'hooks': [
            {'type': 'EvaluationHook'},
            {
                'type': 'BestCkptSaverHook',
                'save_file_name': 'best_model.pth',
                'metric_key': 'f1',
                'save_optimizer': False,
                'restore_best': True,
            },
            {'type': 'TextLoggerHook', 'interval': 50},
            {'type': 'IterTimerHook'},
        ],
    },
    'evaluation': {'dataloader': {'workers_per_gpu': 0, 'shuffle': False}},
}
