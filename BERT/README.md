# BERT Model for Natural Language Processing - Google Research  

Tutorial at: https://www.linkedin.com/feed/update/urn:li:article:7199689357226496574/  

```
$ ctpu up

$ ctpu status

$ ctpu ls

$ pip install pandas sklearn scikit-learn

$ sudo chmod -R 777 /home

$ mkdir BERT 

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

$$ python run_classifier.py --task_name=cola --bert_config_file=gs://tpu22/bert_config.json --vocab_file=gs://tpu22/vocab.txt --init_checkpoint=gs://tpu22/bert_model.ckpt --data_dir=/home/BERT/data --output_dir=gs://tpu22/tpu-output --do_lower_case=True --max_seq_length=400 --do_train=True --do_eval=True --do_predict=True --train_batch_size=128 --eval_batch_size=128 --predict_batch_size=128 --learning_rate=2e-6 --num_train_epochs=5.0 --warmup_proportion=0.1 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8 --tpu_name=rubens  
```


<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert1.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert00.JPG>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert05.JPG>  
