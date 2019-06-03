# BERT Model for Natural Language Processing - Google Research  

Tutorial at: https://www.linkedin.com/feed/update/urn:li:article:7199689357226496574/  

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

$ wget https://he-s3.s3.amazonaws.com/media/hackathon/predict-the-happiness/predict-the-happiness/f2c2f440-8-dataset_he.zip

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
--vocab_file=gs://tpu22/vocab.txt --init_checkpoint=gs://tpu22/bert_model.ckpt --data_dir=/home/BERT/data  
--output_dir=gs://tpu22/tpu-output --do_lower_case=True --max_seq_length=400 --do_train=True --do_eval=True  
--do_predict=True --train_batch_size=128 --eval_batch_size=128 --predict_batch_size=128 --learning_rate=2e-6 
--num_train_epochs=5.0 --warmup_proportion=0.1 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000  
--num_tpu_cores=8 --tpu_name=rubens    
```


<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert1.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert00.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert05.JPG>  

# Fine-Tuning BERT Large  

```
$ python run_classifier.py --task_name=cola --bert_config_file=gs://tpu-large/bert_config.json   
--vocab_file=gs://tpu-large/vocab.txt --init_checkpoint=gs://tpu-large/bert_model.ckpt --data_dir=/home/BERT/data  
--output_dir=gs://tpu-large/tpu-output --do_lower_case=True --max_seq_length=400 --do_train=True --do_eval=True   
--do_predict=False --train_batch_size=32 --eval_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5   
--num_train_epochs=3.0 --warmup_proportion=0.1 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000  
--num_tpu_cores=8 --tpu_name=rubens  
```  

<img src=https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/BERT/Pics/large.png>  

<img src=https://raw.githubusercontent.com/RubensZimbres/Repo-2019/master/BERT/Pics/3epochs_Large.png>

# SQuAD Fine Tuning - Question Pairs

```
$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

$ wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py

$ python run_squad.py --bert_config_file=gs://tpu-large/bert_config.json --vocab_file=gs://tpu-large/vocab.txt  
--init_checkpoint=gs://tpu-large/bert_model.ckpt --data_dir=/home/data --output_dir=gs://tpu-squad/tpu-squad-output  
--max_seq_length=384 --do_train=True --do_predict=True --train_batch_size=24 --predict_batch_size=24 --learning_rate=1e-6  
--num_train_epochs=2.0 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8  
--train_file=/home/train-v1.1.json --predict_file=/home/dev-v1.1.json --doc_stride=128 --tpu_name=rubens

$ python evaluate-v1.1.py dev-v1.1.json /home/squad/predictions.json
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/squad_both.png>  

<b>SQuAD Training</b>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/squad_training2.png>  

<b>Restoring checkpoints from training</b>  

```
$ python run_squad.py --bert_config_file=gs://tpu-large/bert_config.json --vocab_file=gs://tpu-large/vocab.txt  
--init_checkpoint=gs://tpu-squad/tpu-squad-output/model.ckpt-5000.index --data_dir=/home/data  
--output_dir=gs://tpu-squad/tpu-squad-output --max_seq_length=384 --do_train=False --do_predict=True  
--train_batch_size=24 --predict_batch_size=24 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000  
--num_tpu_cores=8 --tpu_name=rubens --train_file=/home/train-v1.1.json --predict_file=/home/dev-v1.1.json  
--doc_stride=128

$ gsutil cp gs://tpu-squad/tpu-squad-output/predictions.json predictions.json

$ python /home/evaluate-v1.1.py /home/dev-v1.1.json predictions.json

```
