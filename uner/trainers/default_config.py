DEFAULT_CONFIG = {
    'train': {
        'hooks': [{
            'type': 'EvaluationHook'
        }, {
            'type': 'CheckpointHook'
        }, {
            'type': 'MyBestCkptSaverHook',
            'metric_key': 'f1'
        }, {
            'type': 'TextLoggerHook',
            'interval': 50
        }, {
            'type': 'IterTimerHook'
        }]
    }
}
