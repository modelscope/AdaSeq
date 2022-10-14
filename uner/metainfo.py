class Tasks:
    named_entity_recognition = 'named-entity-recognition'


class Models:
    sequence_labeling_model = 'sequence-labeling-model'
    global_pointer_model = 'global-pointer-model'


class Metrics:
    sequence_labeling_metric = 'sequence-labeling-metric'
    ner_metric = 'ner-metric'
    span_extraction_metric = 'span-extraction-metric'


class Preprocessors:
    nlp_preprocessor = 'nlp-preprocessor'
    sequence_labeling_preprocessor = 'sequence-labeling-preprocessor'
    global_pointer_preprocessor = 'global-pointer-preprocessor'


class Trainers:
    default_trainer = 'default-trainer'
    ner_trainer = 'ner-trainer'


class Optimizers:
    pass


class LR_Schedulers:
    pass


class Hooks:
    pass


class DatasetDumpers:
    ner_dumper = 'ner-dumper'
