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
    sample_rate_hertz=8000,enable_word_time_offsets= True,
    language_code='pt-BR',
    enable_automatic_punctuation= True,
    use_enhanced=True,
    speech_contexts=[speech.types.SpeechContext(phrases=['comunicação', 'corporativa','smarphone'])],
    enable_speaker_diarization=True,
    diarization_speaker_count=2,
    audio_channel_count=1,
    profanity_filter=True,
    enable_separate_recognition_per_channel=False)

audio=speech.types.RecognitionAudio(uri="gs://audios/Gisele0_3people.wav")

operation = client.long_running_recognize(config, audio)

print('Waiting for operation to complete...')
response = operation.result(timeout=10000000)
response

texto=[]



for i in range(0,len(response.results)):
    print(response.results[i].alternatives[0].transcript)
    texto.append(response.results[i].alternatives[0].transcript)

texto

with open('/home/rubens/Documents/Transcricao_Gisele_0.txt', 'w') as f:
    for item in texto:
        f.write("%s\n" % item)
