import os
import sys
import collections
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import time

os.listdir("/home/rubensvectomobile/BERT/bert")
sys.path.insert(0, '/home/rubensvectomobile/BERT/bert')

from run_classifier import *
import modeling
import optimization
import tokenization

task_name='cola' 
bert_config_file='gs://tpu22/bert_config.json'
vocab_file='gs://tpu22/vocab.txt'
init_checkpoint='gs://tpu22/bert_model.ckpt'
data_dir='/home/BERT/rubensvectomobile/data'
output_dir='gs://tpu22/tpu-output33'
do_lower_case=True 
max_seq_length=384 
do_train=True 
do_eval=True 
do_predict=True
train_batch_size=24
eval_batch_size=24
predict_batch_size=24
learning_rate=1.4e-5
num_train_epochs=7.0
warmup_proportion=0.1
use_tpu=True
save_checkpoints_steps=1 
iterations_per_loop=1000 
num_tpu_cores=8 
tpu_name='rubensvectomobile'

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

processor = ColaProcessor()
label_list = processor.get_labels()

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tpu_cluster_resolver = None
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
  cluster=tpu_cluster_resolver,
  master='grpc://TPU_IP:8470',
  model_dir=output_dir,
  save_checkpoints_steps=save_checkpoints_steps,
  tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=iterations_per_loop,
      num_shards=num_tpu_cores,
      per_host_input_for_training=is_per_host))

train_examples = processor.get_train_examples(data_dir)
num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)

model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu)

estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=train_batch_size)
      
      
train_file = os.path.join(output_dir, "train.tf_record")

file_based_convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer, train_file)

tf.logging.info("***** Running training *****")
tf.logging.info("  Num examples = %d", len(train_examples))
tf.logging.info("  Batch size = %d", train_batch_size)
tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = file_based_input_fn_builder(
    input_file=train_file,
    seq_length=max_seq_length,
    is_training=True,
    drop_remainder=True)
    
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
