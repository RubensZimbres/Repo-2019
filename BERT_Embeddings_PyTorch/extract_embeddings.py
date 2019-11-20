import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
import numpy as np
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
from transformers import *

class Configuracao(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


torch.cuda.is_available()

config = Configuracao(
    testing=False,
    bert_model_name="bert-base-multilingual-uncased",
    max_lr=3e-5,
    epochs=12,
    use_fp16=True,
    bs=64,
    discriminative=False,
    max_seq_len=128,
)

from pytorch_pretrained_bert import BertTokenizer

bert_token = BertTokenizer.from_pretrained(
    config.bert_model_name,
)

class FastAiBertTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, *args, **kwargs):
        return self
    def tokenizer(self, t:str) -> List[str]:
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]

fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_token, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])

fastai_bert_vocab = Vocab(list(bert_token.vocab.keys()))

print(fastai_bert_vocab)

import pandas as pd
alertas_atendente=pd.read_csv('gs://fast-ai-gif/atendente_words_1_test_ok.csv',sep=',')

print(alertas_atendente.iloc[:,0])

alertas_atendente.iloc[:,0].to_csv('alertas_embed.txt', index=None,sep=',')

from pytorch_pretrained_bert import BertModel
model = BertModel.from_pretrained('bert-base-multilingual-uncased')
print(model.embeddings.word_embeddings)

import extract_direto as extrair

embeddings=extrair.Main()
embeddings.main(input_file='alertas_embed.txt',output_file='/home/rubensvectomobile_gmail_com/alertas2.json')

###### OR

#import os
#os.system('python3 extract.py --input_file=alertas_embed.txt --output_file=/home/rubensvectomobile_gmail_com/alertas.json --bert_model=bert-base-multilingual-uncased --max_seq_length=128 --batch_size=16')
