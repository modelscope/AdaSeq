from abs import ABC, abstractmethod

class CorpusReader(ABC):

    @abstractmethod
    def load_data_file(cls, file_path, corpus_config):
        raise NotImplementedError

class SequenceLabelingCorpusReader(CorpusReader):

    def load_data_file(cls, file_path, corpus_config):
        if corpus_config['format'] == 'column':
            return self.load_column_data_file(file_path, delimiter = corpus_config.get('delimiter', '\t'))
        elif corpus_config['format'] == 'json':
            return self.load_sequence_labeling_json_data_file(file_path, token)

    def load_column_data_file(cls, file_path, delimiter):
        with open(file_path, encoding='utf-8') as f:
            guid = 0
            tokens = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if tokens:
                        spans = self.labels_to_spans(labels)
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'spans': spans
                        }
                        guid += 1
                        tokens = []
                        labels = []
                else:
                    splits = line.split(delimiter)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                spans = self.labels_to_spans(labels)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans}


    def labels_to_spans(cls, labels):
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


class SpanBasedCorpusReader(CorpusReader):

    def load_data_file(cls, file_path, corpus_config):
        
