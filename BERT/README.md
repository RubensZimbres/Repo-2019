# BERT Model for Natural Language Processing - Google Research

```
$ sudo chmod -R 777 /home

$ mkdir BERT  

$ wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

$ git clone https://github.com/google-research/bert.git  

$ wget https://he-s3.s3.amazonaws.com/media/hackathon/predict-the-happiness/predict-the-happiness/f2c2f440-8-dataset_he.zip

$ python format.py
```  

<b>Folder Structure</b>  

-|-bert  
-|-bert_output  
-|-data  
-|-multi_cased_L-12_H-768_A-12  


```
$ export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12

$ python run_classifier.py --task_name=cola --do_train=false -–do_eval=true --do_predict=true --data_dir=/home/BERT/data/ --vocab_file=/home/BERT/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/BERT/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/BERT/multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=100 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/home/BERT/bert_output/ --do_lower_case=False

$ try $ ython run_classifier.py --task_name=cola --do_train=False -–do_eval=True --do_predict=True --data_dir=/home/BERT/data --vocab_file=/home/BERT/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/BERT/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12/bert_model.ckpt --task_name=cola --max_seq_length=100  --output_dir=/home/BERT/bert_output/ --load_all_detection_checkpoint_vars: true  --do_lower_case=False

```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert0.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert1.png>  
