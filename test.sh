export PYTHONPATH=/home/huaijian/caffe-jacinto/python:$PYTHONPATH

cd ../caffe-jacinto/
make -j20
cd ../paper/

python visible_ssd.py
