# Copyright (c) Alibaba, Inc. and its affiliates.
# Some codes are modified from AllenNLP.
# Copyright (c) AI2 AllenNLP. Licensed under the Apache License, Version 2.0.

from typing import Dict, List, Optional, Tuple

TypedSpan = Tuple[int, int, str]


class InvalidTagSequence(Exception):  # noqa: D101
    def __init__(self, tags=None):
        super().__init__()
        self.tags = tags or list()

    def __str__(self):
        return ' '.join(self.tags)


def bio_tags_to_spans(tags: List[str], classes_to_ignore: Optional[List[str]] = None) -> List[Dict]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Span start is inclusive (closed), end is exclusive (open).
    This function works properly when the spans are unlabeled (i.e., your labels
    are simply "B", "I", and "O").
    # Parameters
    tags : `List[str]`, required.
        The string class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[Dict]`
        The typed, extracted spans from the sequence, in the format
        `{'start': start, 'end': end, 'type': label}`.
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or list()
    spans: List[Dict] = list()
    start = 0
    end = 0
    active_category = None
    for index, string_tag in enumerate(tags):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in 'BIO':
            raise InvalidTagSequence(tags)
        category = string_tag[2:]
        if bio_tag == 'O' or category in classes_to_ignore:
            # The span has ended.
            if active_category is not None:
                spans.append(dict(start=start, end=end + 1, type=active_category))
            active_category = None
            continue
        elif bio_tag == 'B':
            # We are entering a new span; reset indices and active tag to new span.
            if active_category is not None:
                spans.append(dict(start=start, end=end + 1, type=active_category))
            active_category = category
            start = index
            end = index
        elif bio_tag == 'I' and category == active_category:
            # We're inside a span.
            end += 1
        else:
            raise InvalidTagSequence(tags)
    # Last token might have been a part of a valid span.
    if active_category is not None:
        spans.append(dict(start=start, end=end + 1, type=active_category))
    return spans
