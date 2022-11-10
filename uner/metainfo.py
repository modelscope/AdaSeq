class Metrics:
    """ Names for different metrics
    """
    ner_metric = 'ner-metric'  # alias of sequence-labeling-metric
    sequence_labeling_metric = 'sequence-labeling-metric'
    span_extraction_metric = 'span-extraction-metric'


class Models:
    """ Names for different models
    """
    sequence_labeling_model = 'sequence-labeling-model'
    global_pointer_model = 'global-pointer-model'
    multilabel_span_typing_model = 'multilabel-span-typing-model'


class Encoders:
    """ Names for different encoders
    """
    span_encoder = 'span-encoder'


class Decoders:
    """ Names for different decoders
    """
    crf = 'crf'
    partial_crf = 'partial-crf'


class Preprocessors:
    """ Names for different preprocessors
    """
    nlp_preprocessor = 'nlp-preprocessor'
    sequence_labeling_preprocessor = 'sequence-labeling-preprocessor'
    global_pointer_preprocessor = 'global-pointer-preprocessor'
    multilabel_span_typing_preprocessor = 'multilabel-span-typing-preprocessor'


class DataCollators:
    """ Names for different data_collators
    """
    data_collator_with_padding = 'DataCollatorWithPadding'
    sequence_labeling_data_collator = 'SequenceLabelingDataCollatorWithPadding'
    span_extraction_data_collator = 'SpanExtractionDataCollatorWithPadding'
    multi_label_span_typing_data_collator = 'MultiLabelSpanTypingDataCollatorWithPadding'


class Trainers:
    """ Names for different trainers
    """
    default_trainer = 'default-trainer'
    ner_trainer = 'ner-trainer'
    typing_trainer = 'typing-trainer'


class Optimizers:
    """ Names for different optimizers
    """
    pass


class LR_Schedulers:
    """ Names for different lr_schedulers
    """
    pass


class Hooks:
    """ Names for different hooks
    """
    pass


class DatasetDumpers:
    """ Names for different dataset dumpers
    """
    ner_dumper = 'ner-dumper'
