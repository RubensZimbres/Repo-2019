from PIL import Image
import pytesseract
import argparse
import cv2
import os

im = cv2.imread('/home/rubens/deeplearningbook.png', cv2.IMREAD_COLOR)
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(im, config=config)
 
print(text)

filename='/home/rubens/test1.tiff'

text00 = pytesseract.image_to_string(Image.open(filename))

print(text[0:1000])
 
#cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)

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
