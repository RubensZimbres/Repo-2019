import json
import os
texto=[]
dirs = os.listdir('/home/rubens')
for file in dirs:
    print(file)
    with open('/home/rubens/'+file) as json_file:
        data = json.loads(json_file.read())
        texto.append(data)

for parte in range(0,len(texto)):
    print([texto[parte]['results'][x]['alternatives'][0]['transcript'] for x in range(0,len(texto[parte]['results']))])
