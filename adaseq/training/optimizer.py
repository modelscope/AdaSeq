# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import build_from_cfg, default_group
from torch import nn

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, cfg: ConfigDict, default_args: dict = None):
    """Build optimizer from config"""
    cfg = cfg.copy()  # proctect the base config is not modified

    if hasattr(model, 'module'):
        model = model.module  # type ignore

    # build parameter groups with different kwargs
    groups = cfg.pop('param_groups', None)
    param_groups = make_parameter_groups(model.named_parameters(), groups)  # type: ignore

    if default_args is None:
        default_args = {}
    default_args['params'] = param_groups

    return build_from_cfg(cfg, OPTIMIZERS, group_key=default_group, default_args=default_args)


# The implementation is adopted from AllenNLP and modified.
# https://github.com/allenai/allennlp/blob/main/allennlp/training/optimizers.py
# Licensed under the Apache License, Version 2.0.
def make_parameter_groups(
    model_parameters: List[Tuple[str, nn.Parameter]],  # type: ignore
    groups: Optional[List[Dict[str, Any]]] = None,
) -> Union[List[Dict[str, Any]], List[nn.Parameter]]:  # type: ignore
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.
    `groups` contains something like:
    ```
    [
        { 'regex': "transformer_model", 'lr': 1e-3 },
        { 'regex': ['re1', 're2', 're3'], 'lr': 1e-4 }
    ]
    ```
    All of key-value pairs specified in each of these dictionaries will passed passed as-is
    to the optimizer, with the exception of a dictionaries that specify `requires_grad` to be `False`:
    ```
    [
        ...
        {'regex': 'regex', 'requires_grad': False }
    ]
    ```
    When a parameter group has `{"requires_grad": False}`, the gradient on all matching parameters
    will be disabled and that group will be dropped so that it's not actually passed to the optimizer.
    Ultimately, the return value of this function is in the right format to be passed directly
    as the `params` argument to a pytorch `Optimizer`.
    If there are multiple groups specified, this is a list of dictionaries, where each
    dict contains a "parameter group" and groups specific options, e.g., {'params': [list of
    parameters], 'lr': 1e-3, ...}.  Any config option not specified in the additional options (e.g.
    for the default group) is inherited from the top level arguments given in the constructor.  See:
    <https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options>.  See also our
    `test_optimizer_parameter_groups` test for an example of how this works in this code.
    The dictionary's return type is labeled as `Any`, because it can be a `List[torch.nn.Parameter]`
    (for the "params" key), or anything else (typically a float) for the other keys.
    """
    if groups:
        # reformat eache group to ('regex', {})
        allennlp_groups = list()
        for k, group_regexes in enumerate(groups):
            regexes = group_regexes.pop('regex')
            if isinstance(regexes, str):
                regexes = [regexes]
            if not isinstance(regexes, list):
                raise ValueError(f'Unsopported regex: {regexes}')
            allennlp_groups.append((regexes, group_regexes))
        groups = allennlp_groups

        # In addition to any parameters that match group specific regex,
        # we also need a group for the remaining "default" group.
        # Those will be included in the last entry of parameter_groups.
        parameter_groups: Union[List[Dict[str, Any]], List[nn.Parameter]] = [  # type: ignore
            {'params': []} for _ in range(len(groups) + 1)
        ]
        # add the group specific kwargs
        for k in range(len(groups)):
            parameter_groups[k].update(groups[k][1])  # type: ignore

        regex_use_counts: Dict[str, int] = {}
        parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]
        for name, param in model_parameters:
            # Determine the group for this parameter.
            group_index = None
            for k, group_regexes in enumerate(groups):
                for regex in group_regexes[0]:  # type: ignore
                    if regex not in regex_use_counts:
                        regex_use_counts[regex] = 0
                    if re.search(regex, name):
                        if group_index is not None and group_index != k:
                            raise ValueError(
                                '{} was specified in two separate parameter groups'.format(name)
                            )
                        group_index = k
                        regex_use_counts[regex] += 1

            if group_index is not None:
                parameter_groups[group_index]['params'].append(param)
                parameter_group_names[group_index].add(name)
            else:
                # the default group
                parameter_groups[-1]['params'].append(param)
                parameter_group_names[-1].add(name)

        # find and remove any groups with 'requires_grad = False'
        no_grad_group_indices: List[int] = []
        for k, (names, group) in enumerate(zip(parameter_group_names, parameter_groups)):
            if group.get('requires_grad') is False:
                no_grad_group_indices.append(k)
                logger.info(
                    'Disabling gradient for the following parameters: %s',
                    json.dumps(sorted(names), indent=2),
                )
                for param in group['params']:
                    param.requires_grad_(False)

                # warn about any other unused options in that group.
                unused_options = {
                    key: val for key, val in group.items() if key not in ('params', 'requires_grad')
                }
                if unused_options:
                    logger.warning(
                        'Ignoring unused options %s for %s',
                        unused_options,
                        json.dumps(sorted(names), indent=2),
                    )
        parameter_group_names = [
            names
            for (k, names) in enumerate(parameter_group_names)
            if k not in no_grad_group_indices
        ]
        parameter_groups = [
            group for (k, group) in enumerate(parameter_groups) if k not in no_grad_group_indices
        ]

        # log the remaining parameter groups
        logger.info('Done constructing parameter groups.')
        for k in range(len(parameter_groups)):
            group_options = {
                key: val for key, val in parameter_groups[k].items() if key != 'params'
            }
            name_string = json.dumps(sorted(parameter_group_names[k]), indent=2)
            logger.info('Group %s: %s, %s', k, group_options, name_string)

        # check for unused regex
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning(
                    'When constructing parameter groups, %s does not match any parameter name',
                    regex,
                )
    else:
        parameter_groups = [param for name, param in model_parameters]

    # Log the number of parameters to optimize
    num_parameters = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum(parameter.numel() for parameter in parameter_group['params'])
        else:
            num_parameters += parameter_group.numel()  # type: ignore
    logger.info('Number of trainable parameters: %s', num_parameters)

    # Move the default group to the first, since `modelscope` only log lr of the first group.
    # This is the fastest way I think.
    parameter_groups.reverse()

    return parameter_groups
