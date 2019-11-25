import json
from pprint import pprint
import numpy as np
import pandas as pd

data = [json.loads(line) for line in open('gensim.json', 'r')]

xx=[]
for parte in range(0,len(data)):
    xx.append(np.mean([data[parte]['features'][i]['layers'][0]['values'] for i in range(0,len(data[parte]['features']))],axis=0))

from scipy.spatial.distance import cosine as cos

df=pd.read_csv('gensim.csv',encoding = "latin-1",header=None)
print(df.shape)

print(len(data))

for i in range(0,len(xx)):
    print(np.array(df)[i],cos(xx[3],xx[i]))

