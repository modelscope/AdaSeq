import json
import sys

from test_challenge import load_json_file

MEETING_KEY = 'meeting_key'
KEYWORD = 'key_word'


def get_predictions(cur_tagging):
    j = 0
    predictions = set({})
    word = ''
    for cur_line in cur_tagging:
        if not cur_line:
            continue
        char, label, predict = cur_line.split('\t')
        if predict[0] in 'BSO':
            if word:
                predictions.add(word)
                word = ''
        if predict[0] in 'BIES':
            word += char
    if word:
        predictions.add(word)
    return predictions


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print('Usage: python <pyfile> <data_path> <test_path> <doc_split_path> <outpath>')
        exit(1)

    testdata = load_json_file(sys.argv[1])
    tagging = open(sys.argv[2], 'r').read().split('\n')
    split_doc = load_json_file(sys.argv[3])
    out_path = sys.argv[4]
    output_f = open(out_path, 'w')
    for i, document in enumerate(testdata):
        cur_outputs = tagging[split_doc[i] : split_doc[i + 1]]
        cur_key_words = list(get_predictions(cur_outputs))
        cur_data = json.dumps(
            {MEETING_KEY: document[MEETING_KEY], KEYWORD: cur_key_words}, ensure_ascii=False
        )
        output_f.write(cur_data + '\n')
    output_f.close()
