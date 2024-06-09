# Import The Libraries
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import os
import speech_recognition as sr
import pyttsx3
import pyautogui
import time
import screen_brightness_control as sbc

# Download The Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'best.pt', force_reload=True)

# Detect in Real-time

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Read from a folder
Folder = r'' #pass your folder path
for image_path in glob.glob(os.path.join(Folder, "*.jpg")):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = model(image)
    results.print()
    results.save()

# Read a single image
image = cv2.imread(r"") #pass your image path and the extension
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
results = model(image)
results.print()
results.save()

# Defining the functions 
player = pyttsx3.init()
voices = player.getProperty('voices')
player.setProperty('voice', voices[1].id)

def talk(text):
    player.say(text)
    player.runAndWait()

def shutdown():
    return os.system("shutdown /s /t 1")

def restart():
    return os.system("shutdown /r /t 1")

def logout():
    return os.system("shutdown -l")

# Applying the commands in real-time
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = model(frame)
    String = str(frame)
    detect = np.squeeze(frame.render())
    cv2.imshow('YOLO', detect)
    
    if 'thumbsup' in String:
        pyautogui.press("volumeup")
    elif 'thumbsdown' in String:
        pyautogui.press("volumedown")
    elif 'peace' in String:
        talk("Free Palestine")
    elif 'hi' in String:
        talk('hi')
    elif 'love' in String:
        talk("I love you too")
    elif 'fist' in String:
        break
    elif 'thankyou' in String:
        shutdown()
    elif 'livelong' in String:
        sbc.set_brightness(75)
    elif 'heart' in String:
        sbc.set_brightness(50)
    elif ' down' in String:
        sbc.set_brightness(25)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()