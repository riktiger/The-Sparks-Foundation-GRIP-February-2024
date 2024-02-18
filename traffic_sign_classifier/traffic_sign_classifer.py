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
model=load_model("/home/aritra/Documents/traffic_sign_classifier/traffic_sign_classifier.h5")
signs = ['Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
            'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 
            'Speed limit (100km/h)', 
            'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]

#Traffic Sign Classification Function
def input_img(image):
    #read_input = cv2.imread(image)
    img_fit = cv2.resize(image,(30,30),interpolation = cv2.INTER_AREA)
    rescaled_image= img_fit/255.
    resized_image = rescaled_image.reshape(-1,30,30,3)
    detector = model.predict(resized_image)[0]
    return {signs[i]: float(detector[i])for i in range(43)}

#UI Creation
Traffic_Sign_Classifier=gr.Interface(fn=input_img,
             inputs=gr.Image(height = 500, width = 500),
             outputs=gr.Label(num_top_classes=5))
Traffic_Sign_Classifier.launch(share=True,debug=True)
