import json
import sys

from test_challenge import load_json_file

MEETING_KEY = 'meeting_key'
KEYWORD = 'key_word'


def get_predictions(cur_tagging):
    j = 0
    predictions = set({})
    while j < len(cur_tagging):
        cur_line = cur_tagging[j]
        if not cur_line:
            j += 1
            continue
        char, label, predict = cur_line.split('\t')
        if predict.startswith('B'):
            cur_word = char
            j += 1
            while j < len(cur_tagging):
                cur_line = cur_tagging[j]
                if not cur_line:
                    break
                char, label, predict = cur_line.split('\t')
                if not predict.startswith('I'):
                    break
                else:
                    cur_word += char
                    j += 1
            predictions.add(cur_word)
        else:
            j += 1
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
