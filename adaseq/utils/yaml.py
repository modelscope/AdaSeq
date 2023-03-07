# Copyright (c) Alibaba, Inc. and its affiliates.
# Borrowed from https://github.com/thesimj/envyaml/blob/main/envyaml/envyaml.py

# This file is part of EnvYaml project
# https://github.com/thesimj/envyaml
#
# MIT License
#
# Copyright (c) 2021 Mykola Bubelich

import io
import os
import re

from yaml import safe_load


def read_yaml(file_path, strict=True, separator='|'):
    """read and parse yaml file

    :param str file_path: path to file
    :param dict cfg: configuration variables (environ and .env)
    :param bool strict: strict mode
    :return: dict
    """
    cfg = os.environ.copy()

    # pattern to remove comments
    RE_COMMENTS = re.compile(r'(^#.*\n)', re.MULTILINE | re.UNICODE | re.IGNORECASE)

    # pattern to extract env variables
    RE_PATTERN = re.compile(
        r"(?P<pref>[\"\'])?"
        r'(\$(?:(?P<escaped>(\$|\d+))|'
        r'{(?P<braced>(.*?))(\|(?P<braced_default>.*?))?}|'
        r'(?P<named>[\w\-\.]+)(\|(?P<named_default>.*))?))'
        r"(?P<post>[\"\'])?",
        re.MULTILINE | re.UNICODE | re.IGNORECASE | re.VERBOSE,
    )

    # read and parse files
    with io.open(file_path, encoding='utf8') as f:
        content = f.read()  # type:str

    # remove all comments
    content = RE_COMMENTS.sub('', content)

    # not found variables
    not_found_variables = set()

    # changes dictionary
    replaces = dict()

    shifting = 0

    # iterate over findings
    for entry in RE_PATTERN.finditer(content):
        groups = entry.groupdict()  # type: dict

        # replace
        variable = None
        default = None
        replace = None

        if groups['named']:
            variable = groups['named']
            default = groups['named_default']

        elif groups['braced']:
            variable = groups['braced']
            default = groups['braced_default']

        elif groups['escaped'] and '$' in groups['escaped']:
            span = entry.span()
            content = (
                content[: span[0] + shifting] + groups['escaped'] + content[span[1] + shifting :]
            )
            # Added shifting since every time we update content we are
            # changing the original groups spans
            shifting += len(groups['escaped']) - (span[1] - span[0])

        if variable is not None:
            if variable in cfg:
                replace = cfg[variable]
            elif variable not in cfg and default is not None:
                replace = default
            else:
                not_found_variables.add(variable)

        if replace is not None:
            # build match
            search = '${' if groups['braced'] else '$'
            search += variable
            search += separator + default if default is not None else ''
            search += '}' if groups['braced'] else ''

            # store findings
            replaces[search] = replace

    # strict mode
    if strict and not_found_variables:
        raise ValueError(
            'Strict mode enabled, variables '
            + ', '.join(['$' + v for v in not_found_variables])
            + ' are not defined!'
        )

    # replace finding with there respective values
    for replace in sorted(replaces, reverse=True):
        content = content.replace(replace, replaces[replace])

    # load proper content
    yaml = safe_load(content)

    # if contains somethings
    if yaml and isinstance(yaml, (dict, list)):
        return yaml

    # by default return empty dict
    return {}
