from PIL import Image
import pytesseract
import argparse
import cv2
import os

im = cv2.imread('/home/rubens/deeplearningbook.png', cv2.IMREAD_COLOR)
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(im, config=config)
 
print(text)

text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
 
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
