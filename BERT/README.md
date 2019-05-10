# BERT Model for Natural Language Processing - Google Research

$ mkdir BERT  

$ wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

$ git clone https://github.com/google-research/bert.git  

$ wget https://he-s3.s3.amazonaws.com/media/hackathon/predict-the-happiness/predict-the-happiness/f2c2f440-8-dataset_he.zip

$ python format.py

<b>Folder Structure</b>  

<img src=>

$ python run_classifier.py --task_name=cola --do_train=false -–do_eval=true  
--do_predict=true --data_dir=/home/rubens/anaconda3/BERT/data/ 
--vocab_file=/home/rubens/anaconda3/BERT/multi_cased_L-12_H-768_A-12/vocab.txt 
--bert_config_file=/home/rubens/anaconda3/BERT/multi_cased_L-12_H-768_A-12/bert_config.json 
--init_checkpoint=/home/rubens/anaconda3/BERT/multi_cased_L-12_H-768_A-12/bert_model.ckpt 
--max_seq_length=400 --train_batch_size=8  --learning_rate=2e-5 –num_train_epochs=3.0 
--output_dir=/home/rubens/anaconda3/BERT/bert_output/ --do_lower_case=False

