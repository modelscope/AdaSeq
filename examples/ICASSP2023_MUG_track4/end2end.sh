#  download files
export root_dir=`pwd`
export comp_dir=$root_dir/examples/ICASSP2023_MUG_track4
export yaml=$comp_dir/configs/bert_crf_sbert.yaml
export data_dir=dataset


# download and preprocess data
cd $comp_dir
mkdir -p $data_dir
if [ $1 ]
then
    python download.py $1
else
    echo "please pass your sdk_token, which can be achieved from 'personal center' of MODELSCOPE HOME PAGE. "
    echo "e.g. ./end2end.sh 3af3-faega-geagea"
    exit
fi
# start training
cd $root_dir
echo $root_dir
echo "start training...."
sed -i "s?\${root_dir}?${root_dir}?g" $yaml

python scripts/train.py -c $yaml

export best_path=`find experiments/kpe_* -name best_model.pth | tail -1`
export exp_path=${best_path%/*}
echo "exp path is $exp_path"

# start evaluating on dev...
echo "start evaluating on dev..."
pip install jieba rouge yake
cd $comp_dir
python evaluate_kw.py $data_dir/dev.json ../../$exp_path/pred.txt $data_dir/split_list_dev.json evaluation.log

# output test
echo "start predicting on test set...."
cd ${root_dir}
sed -i 's/_dev/_test/g' $exp_path/config.yaml
python scripts/test.py -w $exp_path
echo "have predicted test file"
mv $exp_path/pred.txt $exp_path/pred_test.txt

cd $comp_dir
python get_keywords.py $data_dir/test.json $root_dir/$exp_path/pred_test.txt $data_dir/split_list_test.json submit.json
echo "predictions on test set has been output to $comp_dir/submit.json"
