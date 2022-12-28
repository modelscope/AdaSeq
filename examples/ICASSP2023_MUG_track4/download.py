import json
import sys

from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from preprocess import process_data


def csv2json(csv_name, out_name='test.json'):
    data_frame = pd.read_csv(csv_name, sep='\t')
    jsons = []
    for i, row in data_frame.iterrows():
        content = json.loads(row['content'])
        jsons.append(content)
    data_dir = csv_name.split('/')[0]
    f = open(f'{data_dir}/{out_name}', 'w')
    json.dump(jsons, f, ensure_ascii=False)
    return jsons


def data2json(data, out_name='test.json'):
    jsons = []
    for eve in data:
        content = json.loads(eve['content'])
        jsons.append(content)
    f = open(out_name, 'w')
    json.dump(jsons, f, ensure_ascii=False)
    return jsons


if __name__ == '__main__':
    api = HubApi()
    sdk_token = sys.argv[1]  # 必填, 从modelscope WEB端个人中心获取
    print('sdk_token is', sdk_token)
    api.login(sdk_token)  # online

    input_config_kwargs = {'delimiter': '\t'}
    data = MsDataset.load(
        'Alimeeting4MUG',
        namespace='modelscope',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        subset_name='default',
        **input_config_kwargs,
    )

    # print(data["test"][0])
    dataset = 'dataset'
    data2json(data['train'], f'{dataset}/train.json')
    data2json(data['validation'], f'{dataset}/dev.json')
    data2json(data['test'], f'{dataset}/test.json')
    test_doc, test = process_data(f'{dataset}/test.json')
    dev_doc, dev = process_data(f'{dataset}/dev.json')
    train_doc, train = process_data(f'{dataset}/train.json')
