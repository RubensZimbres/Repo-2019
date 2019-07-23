from PIL import Image
import pytesseract
import argparse
import os

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy 
import matplotlib.pyplot as plt

im = plt.imread('/home/rubens/file5_deep.tiff')

config = ('-l eng --oem 3 --psm 10')
text00 = pytesseract.image_to_string(im, config=config)
 
print(text)


# ! pip install --upgrade google-cloud-texttospeech

client = texttospeech.TextToSpeechClient()

synthesis_input = texttospeech.types.SynthesisInput(text=text00)

voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

response = client.synthesize_speech(synthesis_input, voice, audio_config)

with open('output.mp3', 'wb') as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
