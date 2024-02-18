# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Libraries
import gradio as gr
import keras
import numpy as np
import requests
import PIL
import cv2
from keras.models import load_model

#Importing the model
model=load_model("/home/aritra/Documents/face_mask_detector/face_mask_detector.h5")
labels = ['No Face Mask Detected',
            'Face Mask Detected']

#Face Mask Detection
def input_img(image):
    img_fit = cv2.resize(image,(64,64),interpolation = cv2.INTER_AREA)
    rescaled_image = img_fit/255.
    resized_image = rescaled_image.reshape(-1,64,64,3)
    y_probability=model.predict(resized_image)
    print(y_probability)
    y_predict = np.where(y_probability > 0.5, 1,0)
    return labels[int(y_predict)]

#UI Creation
Face_Mask_Detector =gr.Interface(fn=input_img,
             inputs=gr.Image(height = 500, width = 500),
             outputs=gr.Textbox(label='Result'))
Face_Mask_Detector.launch(share=True,debug=True)
