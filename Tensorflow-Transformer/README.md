# Tensorflow Transformer  

Replication of code from Attention is All You Need paper

```
$ pip install tf-nightly-gpu
$ git clone https://github.com/tensorflow/models

$ pip install --user -r /home/rubens/models/official/requirements.txt
$ mkdir data

~/anaconda3/models/official/transformer$ cp data_download.py /home/rubens/anaconda3/models/data_download.py

$ cd ~/anaconda3/models

$ python data_download.py --data_dir=/home/rubens/anaconda3/models/data

$ python transformer_main.py --data_dir=/home/rubens/models/official/transformer/data --model_dir=/home/rubens/models/model_big --vocab_file=/home/rubens/models/official/transformer/data/vocab.ende.32768 --param_set=big  --train_steps=20000 --steps_between_evals=1000 --bleu_source=/home/rubens/models/official/transformer/data/newstest2014.en --bleu_ref=/home/rubens/models/official/transformer/data/newstest2014.de 

# Change vocab_size from 33708 to 33945 in model_params.py #4974 (https://github.com/tensorflow/models/issues/4900)

$ vi /models/official/transformer/model/model_params.py

```
