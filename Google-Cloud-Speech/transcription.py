from google.cloud import speech_v1p1beta1 as speech
from google.cloud import speech_v1 as speech
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types

client = speech.SpeechClient()

config = speech.types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    model='command_and_search',
    sample_rate_hertz=8000,
    enable_word_time_offsets= True,
    language_code='pt-BR',
    enable_automatic_punctuation= True,
    use_enhanced=True,
    speech_contexts=[speech.types.SpeechContext(phrases=['sim','não','pacote','linha', 'endereço','data','telefone','senhor','senhora'])],
#    enable_speaker_diarization=False,
#    diarization_speaker_count=2,
    audio_channel_count=2,
    profanity_filter=True,
    enable_separate_recognition_per_channel=True)


audio=speech.types.RecognitionAudio(uri="gs://folder/ggf_output.wav")

operation = client.long_running_recognize(config, audio)

print('Waiting for operation to complete...')

response = operation.result(timeout=10000)

for i in range(0,len(response.results)):
    print('Speaker:',response.results[i].channel_tag,response.results[i].alternatives[0].transcript)
