import json
import os
import re
import sys

from test_challenge import analysis, construct_label, load_json_file

CANDIDATE = 'candidate'
KEY_SENTENCE = 'key_sentence'
KEY_WORD = 'key_word'
SENTENCES = 'sentences'


def get_predictions(cur_tagging, cur_doc):
    j = 0
    predictions = {}
    while j < len(cur_tagging):
        cur_line = cur_tagging[j]
        if not cur_line:
            j += 1
            continue
        char, label, predict = cur_line.split('\t')
        if predict.startswith('B') or predict.startswith('S'):
            cur_word = char
            j += 1
            while j < len(cur_tagging):
                cur_line = cur_tagging[j]
                if not cur_line:
                    break
                char, label, predict = cur_line.split('\t')
                if not (predict.startswith('I') or predict.startswith('E')):
                    break
                else:
                    cur_word += char
                    j += 1
            if cur_word not in predictions:
                nums = len(re.findall(cur_word, cur_doc))
                predictions[cur_word] = nums
        else:
            j += 1
    return predictions


if __name__ == '__main__':
    # if len(sys.argv) != 5:
    #     print('Usage: python <pyfile> <data_path> <test_path> <doc_split_path> <outpath>')
    #     exit(1)
    results = {}
    labels = []
    testdata = load_json_file(sys.argv[1])
    meeting_keys = [x['meeting_key'] for x in testdata]
    tagging = open(sys.argv[2], 'r').read().split('\n')
    split_doc = load_json_file(sys.argv[3])
    assert len(split_doc) == len(testdata) + 1

    top_k = [10, 15, 20]
    results['all'] = []
    for k in top_k:
        results[k] = []
    for i, document in enumerate(testdata):
        cur_outputs = tagging[split_doc[i] : split_doc[i + 1]]
        cur_labels = construct_label(document[CANDIDATE])
        cur_content = ''.join([s['s'] for s in document[SENTENCES]])
        cur_key_words = get_predictions(cur_outputs, cur_content)
        sorted_tuple = sorted(cur_key_words.items(), key=lambda x: x[1], reverse=True)
        results['all'].append(sorted_tuple)
        for k in top_k:
            results[k].append(sorted_tuple[:k])
        labels.append(cur_labels)

    output_path = sys.argv[-1]

    output_dir = '/'.join(output_path.split('/')[:-1])
    # print('analysis all extracted words...')
    # analysis(results['all'], labels, meeting_keys, output_path)
    filename = output_path.split('/')[-1]
    if '.' in filename:
        filename, suffix = filename.split('.')
    else:
        suffix = ''
    for k in top_k:
        cur_name = filename + f'_{k}' + '.' + suffix
        cur_path = os.path.join(output_dir, cur_name)
        print(f'analysis top@{k} words...')
        analysis(results[k], labels, meeting_keys, cur_path)
