import os

import datasets
from datasets import Features, Value


class NamedEntityRecognitionDatasetBuilder(datasets.GeneratorBasedBuilder):

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features({
                'id':
                Value('string'),
                'tokens': [Value('string')],
                'spans': [{
                    'start': Value('int32'),  # close
                    'end': Value('int32'),  # open
                    'type': Value('string')
                }]
            }))
        return info

    def _split_generators(self, dl_manager):
        if self.config.data_files is not None:
            data_files = dl_manager.download_and_extract(
                self.config.data_files)
            if isinstance(data_files, dict):
                return [
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={'filepath': data_files[split_name][0]})
                    for split_name in data_files.keys()
                ]
            elif isinstance(data_files, (str, list, tuple)):
                if isinstance(data_files, str):
                    data_files = [data_files]
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={'filepath': data_files[0]})
                ]
        elif self.config.data_dir is not None:
            data_dir = dl_manager.download_and_extract(self.config.data_dir)
            all_files = os.listdir(data_dir)
            splits = []
            for split_name in ['train', 'valid', 'test']:
                data_file = get_file_by_keyword(all_files, split_name)
                if data_file is None and split_name == 'valid':
                    data_file = get_file_by_keyword(all_files, 'dev')
                if data_file is None:
                    continue
                data_file = os.path.join(data_dir, data_file)
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name, gen_kwargs={'filepath': data_file}))
            return splits
        else:
            raise ValueError('Datasets cannot be resolved!')

    def _generate_examples(self, filepath):
        if filepath.endswith('.json'):
            raise NotImplementedError
        else:
            return load_column_data_file(filepath)


def get_file_by_keyword(files, keyword):
    for filename in files:
        if keyword in filename:
            return filename
    return None


def load_column_data_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        guid = 0
        tokens = []
        labels = []
        for line in f:
            if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                if tokens:
                    spans = labels_to_spans(labels)
                    yield guid, {
                        'id': str(guid),
                        'tokens': tokens,
                        'spans': spans
                    }
                    guid += 1
                    tokens = []
                    labels = []
            else:
                splits = line.split()
                tokens.append(splits[0])
                labels.append(splits[-1].rstrip())
        if tokens:
            spans = labels_to_spans(labels)
            yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans}


def labels_to_spans(labels):
    spans = []
    in_entity = False
    start = -1
    for i in range(len(labels)):
        # fix label error
        if labels[i][0] in 'IE' and not in_entity:
            labels[i] = 'B' + labels[i][1:]
        if labels[i][0] in 'BS':
            if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                start = i
            else:
                spans.append({'start': i, 'end': i + 1, 'type': labels[i][2:]})
        elif labels[i][0] in 'IE':
            if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                assert start >= 0, \
                    'Invalid label sequence found: {}'.format(labels)
                spans.append({
                    'start': start,
                    'end': i + 1,
                    'type': labels[i][2:]
                })
                start = -1
        if labels[i][0] in 'B':
            in_entity = True
        elif labels[i][0] in 'OES':
            in_entity = False
    return spans
