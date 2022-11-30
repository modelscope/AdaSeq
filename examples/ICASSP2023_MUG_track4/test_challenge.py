import codecs
import json
import os
import sys

import jieba

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import time

import yake
from rouge import Rouge

rouge = Rouge()


def getFileList(fpath):
    if not os.path.exists(fpath):
        print('file path not exists: ' + fpath)
        exit(1)
    files = os.listdir(fpath)
    finalsrcfiles = []
    finaltgtfiles = []
    for _f in files:
        if os.path.isdir(fpath + '/' + _f):
            if _f[0] == '.':
                continue
            testFiles = os.listdir(fpath + '/' + _f)
            for _file in testFiles:
                if 'refs' in _file and 'swp' not in _file:
                    finalsrcfiles.append(fpath + '/' + _f + '/' + _file)
                if 'keywords' in _file and 'keywords_result' not in _file:
                    finaltgtfiles.append(fpath + '/' + _f + '/' + _file)

    return finalsrcfiles, finaltgtfiles


def loadFile(fpath):
    try:
        with codecs.open(fpath, 'r', 'utf-8') as ffile:
            content = ffile.read()
            # words = jieba.lcut(content)
            # result = ' '.join(words)
            result = content
    except:
        print('load file error:' + fpath)
        exit(1)
    return result


def writeFile(contents, fpath):
    try:
        with codecs.open(fpath, 'w', 'utf-8') as ffile:
            for line in contents:
                ffile.write(line + '\n')
    except:
        print('write file error:' + fpath)
        exit(1)


def loadTargetFile(fpath):
    keywords = []
    try:
        with codecs.open(fpath, 'r', 'utf-8') as ffile:
            for line in ffile:
                _word = line.strip()
                if len(_word) > 1:
                    keywords.append(_word)
    except:
        print('load file error:' + fpath)
        exit(1)
    return keywords


def load_json_file(fpath):
    with codecs.open(fpath, 'r', 'utf-8') as ffile:
        jsondata = json.load(ffile)
    return jsondata


def construct_label(candidates):
    final_keywords = set()
    for candidate in candidates:
        final_keywords = set(candidate['key_word']) | final_keywords
    return final_keywords


def keywordExtractor(document, pyake, ngram):
    content = '\n'.join([x['s'] for x in document['sentences']])
    result = pyake.extract_keywords(content)
    labels = construct_label(document['candidate'])
    keywords = [kw[0].replace(' ', '') for kw in result]
    return result, labels


def isFuzzyMatch(firststring, secondstring):
    # 判断两个字符串是否模糊匹配;标准是最长公共子串长度是否>=2
    # first_str = unicode(firststring, "utf-8", errors='ignore')
    # second_str = unicode(secondstring, "utf-8", errors='ignore')
    first_str = firststring.strip()
    second_str = secondstring.strip()
    len_1 = len(first_str)
    len_2 = len(second_str)
    len_vv = []
    global_max = 0
    for i in range(len_1 + 1):
        len_vv.append([0] * (len_2 + 1))
    # len_vv = [[0] * (len_2 + 1)] * (len_1 + 1)  # %第一个50为列数，第二个50为行数，这里认为输入字符串的长度不会超过50，存储两个字符串对应位置的LCS长度
    if len_1 == 0 or len_2 == 0:
        return False
    # print ("str1:%s, str2:%s" % (firststring, secondstring))
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if first_str[i - 1] == second_str[j - 1]:  # 根据对应的字符是否相等来判断
                len_vv[i][j] = 1 + len_vv[i - 1][j - 1]  # 长度二维数组的值
            else:
                # len_vv[i][j] = max(len_vv[i - 1][j], len_vv[i][j - 1])
                len_vv[i][j] = 0
            global_max = max(global_max, len_vv[i][j])
    # print ("str1:%s, str2:%s, lcs:%d" % (firststring, secondstring, len_vv[len_1][len_2]))
    # for eve in len_vv:
    #     print(eve)
    # print(len_vv[len_1][len_2])
    # print(global_max)
    # if len_vv[len_1][len_2] >= 2:
    #     return True
    if global_max >= 2:
        return True
    return False


def calculateApproximateMatchScore(keywords, goldenwords):
    recallLength = len(goldenwords)
    precisionLength = len(keywords)
    nonRecall = []
    precisionNum = 0
    recallNum = 0
    for _key in keywords:
        for _goldenkey in goldenwords:
            if isFuzzyMatch(_key, _goldenkey):
                precisionNum += 1
                break
    for _goldenkey in goldenwords:
        flag = False
        for _key in keywords:
            if isFuzzyMatch(_key, _goldenkey):
                flag = True
                recallNum += 1
                break
        if not flag:
            nonRecall.append(_goldenkey)
    precisionScore = float(precisionNum) / float(precisionLength)
    recallScore = float(recallNum) / float(recallLength)
    fScore = 0.0
    if (precisionScore + recallScore) != 0:
        fScore = 2 * precisionScore * recallScore / (precisionScore + recallScore)
    return fScore, nonRecall


def calculateRouge(keywords, goldenwords):
    scores = rouge.get_scores(hyps=keywords, refs=goldenwords)
    return {
        'rouge-1': scores[0]['rouge-1']['f'],
        'rouge-2': scores[0]['rouge-2']['f'],
        'rouge-l': scores[0]['rouge-l']['f'],
    }


def analysis(results, labels, meeting_keys, outpath):
    assert len(results) == len(labels)
    print('test size:', len(results))
    finalRouge1 = 0.0
    finalFuzzyScore = 0.0
    finalLength = 0.0
    documentNum = len(results)
    outfile = codecs.open(outpath, 'w', 'utf-8')
    for _result, _label, _key in zip(results, labels, meeting_keys):
        rouge = calculateRouge(
            ' '.join([kw[0] for kw in _result]), ' '.join([x.lower() for x in _label])
        )
        fuzzyScore, noRecalls = calculateApproximateMatchScore(
            [kw[0] for kw in _result], [x.lower() for x in _label]
        )
        finalLength += len(_label)
        finalRouge1 += rouge['rouge-1']
        finalFuzzyScore += fuzzyScore
        outfile.write('----------------------------\n')
        outfile.write('meeting keys:' + _key + '\n')
        outfile.write('yake result:' + json.dumps(_result, ensure_ascii=False) + '\n')
        outfile.write('failed recall:' + json.dumps(noRecalls, ensure_ascii=False) + '\n')
        outfile.write(
            'f1 score = %.3f, fuzzy f1 score = %.3f' % (rouge['rouge-1'], fuzzyScore) + '\n'
        )
    print('rouge-1 score: ', finalRouge1 / documentNum)
    outfile.write('*********************************\n')
    outfile.write('rouge-1 score: %.3f\n' % (finalRouge1 / documentNum))
    print('fuzzy match score: ', finalFuzzyScore / documentNum)
    outfile.write('fuzzy match score: %.3f\n' % (finalFuzzyScore / documentNum))
    print('avg labels size: ', finalLength / documentNum)
    outfile.write('avg label size: %d\n' % (finalLength / documentNum))
    outfile.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python <pyfile> <inpath> <outpath>')
        exit(1)
    results = []
    labels = []
    testdata = load_json_file(sys.argv[1])
    meeting_keys = [x['meeting_key'] for x in testdata]
    ngram = 3
    topk = 5
    window_size = 1
    pyake = yake.KeywordExtractor(lan='zh', n=ngram, top=topk, windowsSize=window_size)
    for _document in testdata:
        _result, _label = keywordExtractor(_document, pyake, ngram)
        labels.append(_label)
        results.append(_result)
    start_eval = time.time()
    analysis(results, labels, meeting_keys, sys.argv[2])
    end_eval = time.time()
    print('eval use time:', end_eval - start_eval)
