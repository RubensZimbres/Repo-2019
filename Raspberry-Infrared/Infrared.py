import os
import RPi.GPIO as GPIO
from time import sleep
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(5,GPIO.IN)
while True:
    GPIO.output(7, GPIO.HIGH)
