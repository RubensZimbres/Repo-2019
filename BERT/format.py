### ON CLOUD TPU

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
 
df = pd.read_csv("/home/BERT/train.csv")
 
BERT = pd.DataFrame({'user_id':df['User_ID'],
            'label':encoder.fit_transform(df['Is_Response']),
            'alpha':['a']*df.shape[0],
            'text':df['Description'].replace(r'\n',' ',regex=True)})
 
BERT_train, BERT_dev = train_test_split(BERT, test_size=0.02)
 
df_test = pd.read_csv("/home/BERT/test.csv")
BERT_test = pd.DataFrame({'User_ID':df_test['User_ID'],
                 'text':df_test['Description'].replace(r'\n',' ',regex=True)})
 
BERT_train.to_csv('/home/BERT/data/train.tsv', sep='\t', index=False, header=False)
BERT_dev.to_csv('/home/BERT/data/dev.tsv', sep='\t', index=False, header=False)
BERT_test.to_csv('/home/BERT/data/test.tsv', sep='\t', index=False, header=True)
