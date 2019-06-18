# Tensorflow - Transformer Translation Model  

Replication of code from "Attention is All You Need" paper

```
$ pip install tensorflow==1.12.0
$ pip install tensorflow-gpu==1.12.0
$ pip install tf-nightly-gpu
$ git clone https://github.com/tensorflow/models

$ pip install --user -r /home/rubens/models/official/requirements.txt
$ mkdir data

~/anaconda3/models/official/transformer$ cp data_download.py /home/rubens/anaconda3/models/data_download.py

# tar -xvf

~/data$ rm newtest2014.tgz

$ cd ~/anaconda3/models

$ python data_download.py --data_dir=/home/rubens/anaconda3/models/data

$ vi model_params.py

$ python transformer_main.py --data_dir=/home/rubens/models/official/transformer/data --model_dir=/home/rubens/models/model_big --vocab_file=/home/rubens/models/official/transformer/data/vocab.ende.32768 --param_set=big  --train_steps=20000 --steps_between_evals=1000 --bleu_source=/home/rubens/models/official/transformer/data/newstest2014.en --bleu_ref=/home/rubens/models/official/transformer/data/newstest2014.de
```  

# Issues   

https://github.com/tensorflow/models/pull/4974

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Tensorflow-Transformer/Pics/params0.png>  

# Training  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/Tensorflow-Transformer/Pics/training01.png>
