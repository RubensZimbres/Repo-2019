# BERT Model for Natural Language Processing - Google Research  

For more info about the details of the algorithm access:  

https://www.linkedin.com/feed/update/urn:li:article:7199689357226496574/  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert_git.png>  

```
$ ctpu up

$ ctpu status

$ ctpu ls

$ pip install pandas sklearn scikit-learn

$ sudo chmod -R 777 /home

$ mkdir BERT 

$ wget https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/BERT/format.py

BERT$ mkdir data

```
<b>Bert Uncased:</b>  
```
$ wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

```
<b>Bert LARGE:</b>  
```
$ wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip

```
<b>Bert Multilingual:</b>  
```
$ wget https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip

$ git clone https://github.com/google-research/bert.git  

$ sudo apt-get install unzip

$ python format.py
```  

<b>Local Folder Structure</b>  

-|-bert  
-|-data  

<b>GCP Storage Folder Structure</b>  

-|-tpu22  
---|-tpu-output  


<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/ctpu11.png>  

```
$ export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12

$ python run_classifier.py --task_name=cola --bert_config_file=gs://tpu22/bert_config.json  
--vocab_file=gs://tpu22/vocab.txt --init_checkpoint=gs://tpu22/bert_model.ckpt  
--data_dir=/home/rubens/BERT/data --output_dir=gs://tpu22/tpu-output --do_lower_case=True  
--max_seq_length=400 --do_train=True --do_eval=True --do_predict=True --train_batch_size=128  
--eval_batch_size=128 --predict_batch_size=128 --learning_rate=2e-6 --num_train_epochs=5.0  
--warmup_proportion=0.1 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000  
--num_tpu_cores=8 --tpu_name=rubens    
```


<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert1.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert00.JPG>  

# Tokenizer  (CPU)

```
$ from bert import *
$ tokenizer = tokenization.FullTokenizer
# FILE # CLASS # FUNCTION
$ tokenizer = tokenization.FullTokenizer(vocab_file='gs://tpu22/vocab.txt', do_lower_case=True)
$ orig_tokens=tokenizer.tokenize("Alicia is an intelligent girl working at Google  !")
$ bert_tokens = []
  orig_to_tok_map = []

  bert_tokens.append("[CLS]")
  for orig_token in orig_tokens:
      orig_to_tok_map.append(len(bert_tokens))
      bert_tokens.extend(tokenizer.tokenize(orig_token))
  bert_tokens.append("[SEP]")
$ entrada='[CLS] the man went to the [MASK1] .[SEP] he bought a [MASK2] of milk. [SEP]'
$ from __future__ import print_function
  with open("entrada.tsv", "w") as f:
      print (entrada, file=f)
```

# Classification [Mask]  (GPU)

```
$ python run_squad.py --bert_config_file=gs://tpu-large/bert_config.json --vocab_file=gs://tpu-large/vocab.txt  
--init_checkpoint=gs://tpu-large/bert_model.ckpt --data_dir=/home/rubens/BERT/entrada  
--output_dir=gs://tpu-squad/tpu-squad-output/mask --max_seq_length=384 --do_train=False  
--do_predict=True --train_batch_size=24 --predict_batch_size=24 --learning_rate=1.4e-5 --num_train_epochs=2.0  
--use_tpu=False --save_checkpoints_steps=1 --iterations_per_loop=1000  
--train_file=/home/rubens/BERT/entrada/entrada.tsv --predict_file=/home/rubens/BERT/bert/dev-v1.1.json --doc_stride=128
```

# Feature extraction (14 , 4 , 768)  (CPU)

This will create a JSON file (one line per line of input) containing the BERT activations from each Transformer layer specified by layers (-1 is the final hidden layer of the Transformer, etc.)

```
$ pip uninstall tensorflow_estimator
$ pip install -Iv tensorflow_estimator==1.13.0
$ echo 'Alicia Silverstone went to the beach ||| Alicia is an actress' > input.txt

$ python extract_features.py --input_file=input.txt --output_file=/home/rubensvectomobile/BERT/bert/output.json  
--bert_config_file=gs://tpu22/bert_config.json --vocab_file=gs://tpu22/vocab.txt  
--init_checkpoint=gs://tpu22/bert_model.ckpt --layers=-1,-2,-3,-4 --max_seq_length=128 --batch_size=24
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/json_feature.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/features.png>  

# Glue tasks  

https://gluebenchmark.com/tasks

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/glue.png>  

# Sentence (and sentence-pair) classification tasks  

```
$ wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

$ python3 download_glue_data.py --data_dir glue_data --tasks all

$ python run_classifier.py --task_name=MRPC --bert_config_file=gs://tpu22/bert_config.json  
--vocab_file=gs://tpu22/vocab.txt --init_checkpoint=gs://tpu22/bert_model.ckpt  
--data_dir=/home/rubensvectomobile/BERT/bert/glue_data/MRPC --output_dir=gs://tpu22/tpu-output  
--do_lower_case=True --max_seq_length=400 --do_train=True --do_eval=True --do_predict=True --train_batch_size=32  
--eval_batch_size=32 --predict_batch_size=32 --learning_rate=1.4e-5 --num_train_epochs=4.0 --warmup_proportion=0.1  
--use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8 --tpu_name=rubens
```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/MRPC.png>


# Fine-Tuning BERT Large  

```
$ 0.68 $ python run_classifier.py --task_name=cola --bert_config_file=gs://tpu-large/bert_config.json 
--vocab_file=gs://tpu-large/vocab.txt --init_checkpoint=gs://tpu-large/bert_model.ckpt --data_dir=/home/rubensvectomobile/BERT/data --output_dir=gs://tpu-large/tpu-output --do_lower_case=True  
--max_seq_length=400 --do_train=True --do_eval=True --do_predict=False --train_batch_size=32 --eval_batch_size=32  
--predict_batch_size=32 --learning_rate=3e-5 --num_train_epochs=14.0 --warmup_proportion=0.1 --use_tpu=True  
--save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8 --tpu_name=rubens
```  

<img src=https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/BERT/Pics/large22.png>  

<img src=https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/BERT/Pics/3epochs_Large22.png>

# SQuAD Fine Tuning - Question Pairs

```
$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

$ wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py

# --learning_rate=1.4e-5 BEST

$$ python run_squad.py --bert_config_file=gs://tpu-large/bert_config.json --vocab_file=gs://tpu-large/vocab.txt  
--init_checkpoint=gs://tpu-large/bert_model.ckpt --data_dir=/home/data --output_dir=gs://tpu-squad/tpu-squad-output  
--max_seq_length=384 --do_train=True --do_predict=True --train_batch_size=24 --predict_batch_size=24 --learning_rate=1.4e-5  
--num_train_epochs=2.0 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8  
--train_file=/home/train-v1.1.json --predict_file=/home/dev-v1.1.json --doc_stride=128 --tpu_name=rubens

$ python evaluate-v1.1.py dev-v1.1.json /home/squad/predictions.json
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/squad_both.png>  

<b>SQuAD Training</b>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/squad_training2.png>  

<b>Restoring checkpoints from training</b>  

```
$ 0.90 $ python run_squad.py --bert_config_file=gs://tpu-large/bert_config.json --vocab_file=gs://tpu-large/vocab.txt  
--init_checkpoint=gs://tpu-squad/tpu-squad-output/model.ckpt-5000.index --data_dir=/home/data  
--output_dir=gs://tpu-squad/tpu-squad-output --max_seq_length=384 --do_train=False --do_predict=True  
--train_batch_size=24 --predict_batch_size=24 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000  
--num_tpu_cores=8 --tpu_name=rubens --train_file=/home/train-v1.1.json --predict_file=/home/dev-v1.1.json  
--doc_stride=128

$ gsutil cp gs://tpu-squad/tpu-squad-output/predictions.json predictions.json

$ python /home/evaluate-v1.1.py /home/dev-v1.1.json predictions.json

```  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/final_squad1.png>  

```
$ gcloud config set project machinelearning-12345  

$ gcloud compute config-ssh  

$ ssh username.us-central1-b.machinelearning-12345

```

