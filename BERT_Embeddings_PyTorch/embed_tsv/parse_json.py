import json
from pprint import pprint
import numpy as np

data = [json.loads(line) for line in open('texto3.json', 'r')]

xx=[]
for parte in range(0,len(data)):
    xx.append(np.mean([data[parte]['features'][i]['layers'][0]['values'] for i in range(0,len(data[parte]['features']))],axis=0))

print(len(xx[0]))

#linex_index": 0, "features":
#{"token": "uma", "layers": [{"index": -1, "values": 
