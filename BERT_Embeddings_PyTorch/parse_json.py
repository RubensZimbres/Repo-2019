import json
from pprint import pprint
import numpy as np

with open('texto.json') as f:
    data = json.load(f)
palavras=6

print(len(data['features']))
for i in range(0,palavras+1):
    print(data['features'][i]['token'],np.mean([data['features'][i]['layers'][x]['values'] for x in range(0,4)],axis=0))
print(len(data['features'][i]['token']))

#linex_index": 0, "features":
#{"token": "uma", "layers": [{"index": -1, "values": 
