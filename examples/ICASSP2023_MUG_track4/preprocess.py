import json
import re
from functools import cmp_to_key

import pandas as pd

LENGTH_LIMIT = 150


def cmp_by_len(x, y):
    a = len(x)
    b = len(y)
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


# print(sorted(['av', 'cda', 'a', 'mlafwe', 'wet'], key=cmp_to_key(cmp_by_len)))

keys = [
    'meeting_key',
    'topic_segment_ids',
    'candidate',
    'sentences',
    'paragraph_segment_ids',
    'action_ids',
]
CANDIDATE = 'candidate'
KEY_SENTENCE = 'key_sentence'
KEY_WORD = 'key_word'
SENTENCES = 'sentences'


def process_data(filename, all_sentences=True):
    final_docs = []
    data = json.loads(open(filename, 'r').read())
    split_list = [0]
    for cur_data in data:
        candidate = cur_data[CANDIDATE]
        sentences = cur_data[SENTENCES]
        words = set({})
        sentence_ids = set({})
        for eve in candidate:
            # print(eve)
            words.update(eve[KEY_WORD])
            sentence_ids.update(eve[KEY_SENTENCE])
        words = sorted(words, key=cmp_to_key(cmp_by_len))
        line_num = 0

        sentences = [s['s'] for s in sentences]
        formatted_s = ''
        cache = ''
        cache_size = 0
        for sentence in sentences:
            cur_length = len(sentence)
            line_num += cur_length
            assert cur_length <= LENGTH_LIMIT
            label = ['O'] * cur_length
            if cur_length + cache_size > LENGTH_LIMIT:
                formatted_s += cache + '\n'
                cache = ''
                cache_size = 0
                line_num += 1
            for word in words:
                iters = re.finditer(word, sentence)
                for it in iters:
                    start, end = it.start(), it.end()
                    label[start] = 'B-KEY'
                    for jj in range(start + 1, end):
                        label[jj] = 'I-KEY'
            for s, l in zip(sentence, label):
                cache += s + '\t' + l + '\n'
            cache_size += cur_length
        # if not cache_size <= LENGTH_LIMIT:
        #     print(cache_size)
        assert cache_size <= LENGTH_LIMIT
        assert cache
        line_num += 1
        formatted_s += cache + '\n'
        cur_doc = formatted_s
        split_list.append(split_list[-1] + line_num)
        final_docs.append(cur_doc)
    final_docs = ''.join(final_docs)
    data_dir, name = filename.split('/')[0:-1], filename.split('/')[-1]
    data_dir = '/'.join(data_dir)
    open('{}/processed_{}.txt'.format(data_dir, name), 'w').write(final_docs)
    open('{}/split_list_{}'.format(data_dir, name), 'w').write(str(split_list))
    return final_docs, split_list


def csv2json(csv_name, out_name='test.json'):
    data_frame = pd.read_csv(csv_name, sep='\t')
    jsons = []
    for i, row in data_frame.iterrows():
        content = json.loads(row['content'])
        jsons.append(content)
    data_dir = csv_name.split('/')[0]
    f = open(f'{data_dir}/{out_name}', 'w')
    json.dump(jsons, f)
    return jsons


if __name__ == '__main__':

    # train = json.loads(open('zyp_data/train.json', 'r').read())
    # dev = open('zyp_data/dev.json', 'r').read()
    dataset = 'dataset'
    dev_doc, dev = process_data(f'{dataset}/dev.json')
    train_doc, train = process_data(f'{dataset}/train.json')
    test_json = csv2json(f'{dataset}/except_TS_test1_without_label.csv')
    test_doc, test = process_data(f'{dataset}/test.json')
