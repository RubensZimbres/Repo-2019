import numpy as np
from google.cloud import datastore

datastore_client = datastore.Client()

kind = 'Task'

for i in range(0,len(df_Palavras['c0_presente'])):
    name = str(df_Palavras['c0_presente'][i])
    task_key = datastore_client.key(kind, name)

    task = datastore.Entity(key=task_key)
    task['atendente'] = [str(df_Palavras['c0_preco'][i]),str(df_Palavras['c0_recomend'][i])]
    task['cliente'] = str(df_Palavras['c0_recomend'][i])

    datastore_client.put(task)

print('Saved {}: {}'.format(task.key.name, task['atendente'], task['cliente']))
