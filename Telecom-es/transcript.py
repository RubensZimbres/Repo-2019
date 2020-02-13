import time
start=time.time()

import io
import os

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

#! pip3 install --upgrade google-cloud-storage
#! pip3 install webapp2
#! pip3 install cloudstorage
#! pip3 install GoogleAppEngineCloudStorageClient
#! pip3 install google-cloud-speech

from google.cloud import speech_v1p1beta1 as speech

client = speech.SpeechClient()

config = speech.types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    #sample_rate_hertz=44100,
    model='command_and_search',
    enable_word_time_offsets= False,
    language_code='es-ES',
    enable_automatic_punctuation= True,
    use_enhanced=True,
    #speech_contexts=[speech.types.SpeechContext(phrases=['um', 'dois','três'])],
    enable_speaker_diarization=True,
    diarization_speaker_count=2,
    audio_channel_count=1,
    profanity_filter=True,
    enable_separate_recognition_per_channel=False)

audio=speech.types.RecognitionAudio(uri="gs://temp/audio_latcom3.wav")

operation = client.long_running_recognize(config, audio)

print('Waiting for operation to complete...')
response = operation.result(timeout=100000000)



from google.protobuf.json_format import MessageToJson
serialized = MessageToJson(response)


with open("/home/rubens/Documents/response2.txt", "w") as text_file:
    text_file.write(serialized)


import json
with open('/home/rubens/Documents/data4.json', 'w') as f:
    json.dump(serialized,f)


texto=[]

for i in range(0,len(response.results)):
    print(response.results[i].alternatives[0].words)
    texto.append(response.results[i].alternatives[0].transcript)

texto


import re
pattern = re.compile(r'\d{3,}')

for i in range(0,len(texto)):
    for match in pattern.findall(re.sub('[^A-Za-z0-9]+', '',texto[i])):
        print(match)



with open('/home/rubens/Documents/Transcricao.txt', 'w') as f:
    for item in texto:
        f.write("%s\n" % item)
        

#response.results[0].alternatives[0].transcript

#for result in response.results:
 #       print(u'Transcript: {}'.format(result.alternatives[0].transcript))
  #      print('Confidence: {}'.format(result.alternatives[0].confidence))
   #     print([result.alternatives[0].words.speaker_tag,result.alternatives[0].transcript])
    #    texto.append([result.alternatives[0].words.speaker_tag,result.alternatives[0].transcript])

#print(texto)

        
#result = response.results[-1]

#words_info = result.alternatives[0].words

# Printing out the output:
#for word_info in words_info:
#    print("word: '{}', speaker_tag: {}".format(word_info.word,
#                                               word_info.speaker_tag))
