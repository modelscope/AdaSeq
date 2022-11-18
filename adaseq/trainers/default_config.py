# Copyright (c) Alibaba, Inc. and its affiliates.
DEFAULT_CONFIG = {
    'train': {
        'hooks': [
            {'type': 'EvaluationHook'},
            {'type': 'CheckpointHook'},
            {
                'type': 'BestCkptSaverHook',
                'metric_key': 'f1',
                'save_optimizer': False,
                'restore_best': True,
            },
            {'type': 'TextLoggerHook', 'interval': 50},
            {'type': 'IterTimerHook'},
        ]
    }
}
