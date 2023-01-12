# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.pretraining_preprocessor)
class PretrainingPreprocessor(NLPPreprocessor):
    """Preprocessor for pretraining.
    span targets are processed into mention_boundary, type_ids, ident_ids.
    examples:
        span: {'start':1, 'end':2, 'type': ['PER']}
        processed: {'mention_boundary': [[1], [2]], 'type_ids':[1], 'ident_ids': ['S-SPAN']]}
    """

    def __init__(self, model_dir: str, **kwargs) -> None:

        super().__init__(model_dir, return_offsets=True, **kwargs)

        label_to_id = kwargs.pop('label_to_id', None)
        labels = kwargs.pop('labels', None)
        if label_to_id is not None:
            self.typing_label_to_id = label_to_id
        elif labels is not None:
            self.typing_label_to_id = self.make_label_to_id(labels)
        else:
            raise ValueError('Must have one of `labels` or `label_to_id`')
        self.typing_id_to_label = {
            v: k for k, v in sorted(self.typing_label_to_id.items(), key=lambda x: x[1])
        }
        # make sure they are aligned.
        assert len(self.typing_id_to_label) not in self.typing_id_to_label
        assert len(self.typing_id_to_label) - 1 in self.typing_id_to_label

        self.ident_label_to_id = {'O': 0, 'B-SPAN': 1, 'I-SPAN': 2, 'E-SPAN': 3, 'S-SPAN': 4}
        self.ident_id_to_label = {0: 'O', 1: 'B-SPAN', 2: 'I-SPAN', 3: 'E-SPAN', 4: 'S-SPAN'}

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Pretraining model"""
        prompt_type = None
        if isinstance(data, str):
            prompt_sep_pos = data.find('[SEP]')
            if prompt_sep_pos > 0:
                prompt_type = data[prompt_sep_pos + len('[SEP]') :]
                data = data[:prompt_sep_pos]
            data = {'text': data}
        if isinstance(data, List):
            try:
                prompt_sep_pos = data.index('[SEP]')
            except ValueError:
                prompt_sep_pos = -1
            if prompt_sep_pos > 0:
                prompt_type = data[prompt_sep_pos + 1 :]
                data = data[:prompt_sep_pos]
            data = {'tokens': data}
        output = super().__call__(data)
        if prompt_type is not None:
            self.__generate_prompt_data(data, output, prompt_type)
        return output

    def __generate_prompt_data(
        self,
        data: Union[str, List, Dict],
        output: Dict[str, Any],
        prompt_type: Union[str, List] = None,
    ):
        testing_mode = False
        # collect types
        if prompt_type is None:
            present_types = set([])
            for span in data['spans']:
                present_types.add(span['type'][0])
            # select target type
            all_type = self.typing_label_to_id.keys()
            threshold = 1.0 / len(all_type)
            while len(all_type) > len(present_types):
                tmp_type = random.sample(all_type, 1)[0]
                if tmp_type not in present_types and random.random() <= threshold:
                    present_types.add(tmp_type)
                    break
            selected_type = random.sample(present_types, 1)[0]
            prompt_type = selected_type
        else:
            testing_mode = True
        # construct data
        prompt_token_ids = self.encode_tokens(list(prompt_type))['input_ids']
        # skip [CLS], in inference model, nlp_preprocessor will add batch dimssion
        if testing_mode:
            prompt_input_ids = list(output['tokens']['input_ids'][0]) + prompt_token_ids[1:]
        else:
            prompt_input_ids = output['tokens']['input_ids'] + prompt_token_ids[1:]
        output['prompt_input_ids'] = prompt_input_ids
        output['prompt_input_mask'] = [True] * len(prompt_input_ids)
        if testing_mode:
            output['prompt_input_ids'] = np.expand_dims(np.array(output['prompt_input_ids']), 0)
            output['prompt_input_mask'] = np.expand_dims(
                np.array(output['prompt_input_mask']), False
            )
            return

        prompt_target_label_ids = []
        prompt_target_label_ids.extend(output['ident_ids'])

        for span in data['spans']:
            if span['type'][0] != selected_type:
                for pos in range(span['start'], span['end']):
                    if pos < len(prompt_target_label_ids):
                        prompt_target_label_ids[pos] = self.ident_label_to_id[
                            'O'
                        ]  # mask mention of the other types
        output['prompt_target_label_ids'] = prompt_target_label_ids
        output['prompt_type'] = prompt_type
