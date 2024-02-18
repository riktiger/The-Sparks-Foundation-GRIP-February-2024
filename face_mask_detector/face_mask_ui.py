#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:01:23 2024

@author: aritra
"""

# Importing Libraries
import numpy as np
import keras
import cv2
import dlib

# Loading the Model
model = keras.saving.load_model('/home/aritra/Documents/face_mask_detector/face_mask_detector.h5')


# Detecting Faces
face_cascade = cv2.CascadeClassifier("/home/aritra/Documents/face_mask_detector/haarcascade_frontalface_default.xml")
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('/home/aritra/Documents/face_mask_detector/videoplayback.mp4')

while True: 
          ret,frame = cap.read()
          gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

          faces = face_cascade.detectMultiScale(gray, 1.2, 6)
          for (x,y,w,h) in faces:
              
              # Detecting Face Mask
              extract = gray[y:y+h, x:x+w]
              rgb = cv2.cvtColor(extract, cv2.COLOR_GRAY2RGB)
              resized_img = cv2.resize(rgb,(64,64))
              rescaled_image = resized_img /255.
              img_array = np.reshape(rescaled_image,(-1,64,64,3))
              prediction = float(model.predict(img_array))
              if prediction > 0.5 :
                  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                  cv2.putText(frame, ("FACE MASK DETECTED   " + str(np.round(prediction*100,2))),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
              elif prediction < 0.5 :
                  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                  cv2.putText(frame, ("FACE MASK NOT DETECTED   " + str(np.round(prediction*100,2))),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
              
          cv2.imshow('UI',frame)
          if cv2.waitKey(1) & 0xFF==ord('q'):
              break

cap.release()
cv2.destroyAllWindows()