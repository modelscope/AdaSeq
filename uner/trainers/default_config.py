DEFAULT_CONFIG = {
    'train': {
        'hooks': [{
            'type': 'EvaluationHook'
        }, {
            'type': 'CheckpointHook',
            'interval': 10
        }, {
            'type': 'BestCkptSaverHook',
            'metric_key': 'f1'
        }, {
            'type': 'TextLoggerHook',
            'interval': 50
        }, {
            'type': 'IterTimerHook'
        }]
    }
}
