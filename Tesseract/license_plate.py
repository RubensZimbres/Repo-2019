from PIL import Image
import pytesseract
import cv2
from matplotlib import pyplot as plt
import numpy as np

filename = "image00.jpg"
image = cv2.imread(filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)


thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
plt.imshow(thresh,cmap='gray')


print(pytesseract.image_to_string(thresh, lang='eng',
                        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
