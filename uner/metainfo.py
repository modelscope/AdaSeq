class Tasks:
    named_entity_recognition = 'named-entity-recognition'
    entity_tying = 'entity-typing'


class Metrics:
    sequence_labeling_metric = 'sequence-labeling-metric'
    ner_metric = 'ner-metric'
    span_extraction_metric = 'span-extraction-metric'


class Models:
    sequence_labeling_model = 'sequence-labeling-model'
    global_pointer_model = 'global-pointer-model'
    multilabel_span_typing_model = 'multilabel-span-typing-model'


class Encoders:
    span_encoder = 'span-encoder'


class Decoders:
    crf = 'crf'
    partial_crf = 'partial-crf'


class Preprocessors:
    nlp_preprocessor = 'nlp-preprocessor'
    sequence_labeling_preprocessor = 'sequence-labeling-preprocessor'
    global_pointer_preprocessor = 'global-pointer-preprocessor'
    multilabel_span_typing_preprocessor = 'multilabel-span-typing-preprocessor'


class DataCollators:
    data_collator_with_padding = 'DataCollatorWithPadding'
    sequence_labeling_data_collator = 'SequenceLabelingDataCollatorWithPadding'
    span_extraction_data_collator = 'SpanExtractionDataCollatorWithPadding'
    multi_label_span_typing_data_collator = 'MultiLabelSpanTypingDataCollatorWithPadding'


class Trainers:
    default_trainer = 'default-trainer'
    ner_trainer = 'ner-trainer'
    typing_trainer = 'typing-trainer'


class Optimizers:
    pass


class LR_Schedulers:
    pass


class Hooks:
    pass


class DatasetDumpers:
    ner_dumper = 'ner-dumper'
