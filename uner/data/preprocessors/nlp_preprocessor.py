from typing import Any, Dict, List, Union

from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from transformers import AutoTokenizer, BertTokenizer

from uner.metainfo import Preprocessors
from uner.utils.data_utils import gen_label2id


@PREPROCESSORS.register_module(module_name=Preprocessors.nlp_preprocessor)
class NLPPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(**kwargs)

        self.max_length = kwargs.pop('max_length', 512)
        self.add_cls_sep = kwargs.pop('add_cls_sep', True)
        self.return_tokens_or_text = kwargs.pop('return_tokens_or_text', True)
        self.return_attention_mask = kwargs.pop('return_attention_mask', True)
        self.return_emission_mask = kwargs.pop('return_emission_mask', False)
        self.return_offsets_mapping = kwargs.pop('return_offsets_mapping',
                                                 False)
        self.tokenizer = self.build_tokenizer(model_dir)

    def build_tokenizer(self, model_dir):
        if 'word2vec' in model_dir:
            return BertTokenizer.from_pretrained(model_dir)
        elif 'nezha' in model_dir:
            return BertTokenizer.from_pretrained(model_dir)
        else:
            return AutoTokenizer.from_pretrained(model_dir)

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
        mask = data.get('mask', [True] * len(tokens))
        input_ids = []
        emission_mask = []
        offset_mapping = []
        for offset, (token, token_mask) in enumerate(zip(tokens, mask)):
            subtoken_ids = self.tokenizer.encode(
                token, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)
            emission_mask.extend([token_mask]
                                 + [False] * (len(subtoken_ids) - 1))
            offset_mapping.extend([(offset, offset + 1)]
                                  + [(offset + 1, offset + 1)]
                                  * (len(subtoken_ids) - 1))
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
            emission_mask = emission_mask[:self.max_length - 2]
            offset_mapping = offset_mapping[:self.max_length - 2]
        if self.add_cls_sep:
            input_ids = [self.tokenizer.cls_token_id
                         ] + input_ids + [self.tokenizer.sep_token_id]
            emission_mask = [False] + emission_mask + [False]
            offset_mapping = [(0, 0)] + offset_mapping + [(0, 0)]
        attention_mask = [1] * len(input_ids)

        output = {
            'input_ids': input_ids,
        }
        if self.return_tokens_or_text:
            output['tokens'] = tokens
        if self.return_attention_mask:
            output['attention_mask'] = attention_mask
        if self.return_emission_mask:
            output['emission_mask'] = emission_mask
        if self.return_offsets_mapping:
            output['offset_mapping'] = offset_mapping
        return output

    def encode_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def map_label_to_id(self,
                        labels: List[str] = None,
                        label2id: Dict[str, int] = None) -> Dict[str, int]:
        if label2id is not None:
            return label2id
        elif labels is not None:
            return self._label2id(labels)
        else:
            raise ValueError('labels or label2id is needed.')

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return gen_label2id(labels)
