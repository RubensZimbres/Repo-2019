! conda install -c conda-forge transformers

# ADD DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]] 
#after variable logger in /opt/anaconda3/lib/python3.7/site-packages/transformers/modeling_tf_utils.py

import tensorflow as tf
import tensorflow_datasets
from transformers import *

#tf.compat.v1.enable_eager_execution()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
data = tensorflow_datasets.load('glue/mrpc')

train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(10)
valid_dataset = valid_dataset.batch(64)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(train_dataset, epochs=10, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

model.save_pretrained('/home/rubensvectomobile_gmail_com/tf')



pytorch_model = BertForSequenceClassification.from_pretrained('/home/rubensvectomobile_gmail_com/tf',from_tf=True)

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

sentence_0 = "Esta pesquisa foi consistente com suas descobertas."
sentence_1 = "Suas descobertas foram compatíveis com esta pesquisa."
sentence_2 = "Suas descobertas não eram compatíveis com esta pesquisa."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
