# BERT Model for Natural Language Processing - Google Research

```
$ ctpu up

$ ctpu status

$ ctpu ls

$ pip install pandas sklearn scikit-learn

$ sudo chmod -R 777 /home

$ mkdir BERT  

$ wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

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

$$ python run_classifier.py --task_name=cola --bert_config_file=gs://tpu/bert_config.json --vocab_file=gs://tpu/vocab.txt --init_checkpoint=gs://tpu/bert_model.ckpt --data_dir=/home/BERT/data --output_dir=/home/BERT/bert_output --do_lower_case=True --max_seq_length=72 --do_train=True --do_eval=True --do_predict=True --train_batch_size=32 --eval_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --warmup_proportion=0.1 --use_tpu=True --save_checkpoints_steps=1 --iterations_per_loop=1000 --num_tpu_cores=8 --tpu_name=rubensblablabla
```

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert0.png>  

<img src=https://github.com/RubensZimbres/Repo-2019/blob/master/BERT/Pics/bert1.png>  
