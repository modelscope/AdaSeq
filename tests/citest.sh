/home/admin/python37/bin/pip3 install -r ./requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

echo $PWD

pre-commit run -c .pre-commit-config_local.yaml --all-files
if [ $? -ne 0 ]; then
    echo "linter test failed, please run 'pre-commit run --all-files' to check"
    exit -1
fi

PYTHONPATH=. /home/admin/python37/bin/python3 tests/run_tests.py
