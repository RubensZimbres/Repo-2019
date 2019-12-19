### Google Cloud Datastore

# SAVE

import json

data = {}
data=pd.DataFrame.to_dict(df_Palavras)

json_data = json.dumps(data,ensure_ascii=False)
json.loads(json_data)

import codecs, json
with codecs.open('data.json', 'w', 'utf8') as f:
     f.write(json.dumps(data,ensure_ascii=False))

# OPEN

with open('strings.json') as f:
    d = json.load(f)
    print(d)
