# Tensorflow Transformer  

Implementation os Attention is All You Need paper

```
$ pip install tf-nightly-gpu
$ git clone https://github.com/tensorflow/models

$ pip install --user -r /home/rubens/models/official/requirements.txt
$ mkdir data
:~/anaconda3/models/official/transformer$ cp data_download.py /home/rubens/anaconda3/models/data_download.py
$ cd ~/anaconda3/models

$ python data_download.py --data_dir=/home/rubens/anaconda3/models/data

$ python transformer_main.py --data_dir=/home/rubens/anaconda3/models/data --model_dir=/home/rubens/anaconda3/models/checkpoints --vocab_file=/home/rubens/anaconda3/models/data/..... --param_set=big
```
