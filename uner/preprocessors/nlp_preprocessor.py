from typing import Any, Dict, List, Union

from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from transformers import AutoTokenizer, BertTokenizer

from uner.metainfo import Preprocessors


@PREPROCESSORS.register_module(module_name=Preprocessors.nlp_preprocessor)
class NLPPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'lstm' in model_dir:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        elif 'nezha' in model_dir:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.max_length = kwargs.pop('max_length', 512)
        self.add_cls_sep = kwargs.pop('add_cls_sep', True)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        if isinstance(data, str):
            data = {'text': data}
        if isinstance(data, List):
            data = {'tokens': data}
        if 'tokens' in data:
            output = self.encode_tokens(data)
        elif 'text' in data:
            output = self.encode_text(data)
        else:
            raise ValueError('Data sample must have "text" or "tokens" field!')
        return output

    def encode_tokens(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tokens = data['tokens']
        input_ids = []
        emission_mask = []
        offset_mapping = []
        for offset, token in enumerate(tokens):
            subtoken_ids = self.tokenizer.encode(
                token, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)
            offset_mapping.extend([(offset, offset + 1)]
                                  + [(offset + 1, offset + 1)]
                                  * (len(subtoken_ids) - 1))
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
            offset_mapping = offset_mapping[:self.max_length - 2]
        if self.add_cls_sep:
            input_ids = [self.tokenizer.cls_token_id
                         ] + input_ids + [self.tokenizer.sep_token_id]
            offset_mapping = [(0, 0)] + offset_mapping + [(0, 0)]
        attention_mask = [1] * len(input_ids)
        emission_mask = [
            1 if start < end else 0 for start, end in offset_mapping
        ]

        output = {
            'tokens': tokens,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'emission_mask': emission_mask,
            'offset_mapping': offset_mapping
        }
        return output

    def encode_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
