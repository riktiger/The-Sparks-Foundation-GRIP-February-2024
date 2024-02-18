#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 02:42:00 2024

@author: aritra
"""
# THE SPARKS FOUNDATION : GRADUATE ROTATIONAL INTERNSHIP PROGRAM (GRIP), FEBRUARY 2024
# Domain - Computer Vision and Internet of Things

## PROBLEM STATEMENT

## INTERNSHIP TASK 1: Color Identification in Images
#Implement an image color detector which identifies all the colors in an iamge or video

## SOLUTION
## AUTHOR : ARITRA BAG



# Importing Libraries
import sklearn
import gradio as gr
import skimage
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import keras

from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter


# Developing the color detection function
def color_detector(rgb_image, max_colors):
    
    ## Detecting the color names
    resized_image = cv2.resize(rgb_image,(1280,720), interpolation = cv2.INTER_AREA)
    resized_image_array = resized_image.reshape(resized_image.shape[0]*resized_image.shape[1], 3)
    clf_detector = KMeans(n_clusters = int(max_colors))
    color_labels = clf_detector.fit_predict(resized_image_array)
    detected_colors = clf_detector.cluster_centers_
    
    ## Function to convert RGB colors to HexaDecimal
    def RGB2HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    
    ## Calculating the percentage of each color
    color_counts = Counter(color_labels)
    ordered_colors = [detected_colors[i] for i in color_counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in color_counts.keys()]
    rgb_colors = [ordered_colors[i] for i in color_counts.keys()]
    rgb_colors = (np.round(rgb_colors, 0)).astype(int)
        
    ## Combining  HEX and RGB Data
    combined_data = []
    for i in range(len(hex_colors)):
        combined_data.append('HEX CODE = ' +str(hex_colors[i])+ ' RGB CODE = ' + str(rgb_colors[i]))
            
    ## Creating the plot for the color spectrum
    output_plot = plt.figure(figsize = (25,25), dpi = 60)
    plt.rcParams.update({'font.size': 15})
    plt.pie(color_counts.values(), labels = combined_data, colors = hex_colors, autopct='%.2f%%')
    plt.tight_layout()
    
    return output_plot

# Developing the Gradio UI
Color_Detector=gr.Interface(fn=color_detector, inputs = [gr.Image(width = 600, height = 400, label = "Enter Picture"), gr.Textbox(label = "Maximum number of colors to detect")], outputs = gr.Plot(label = " Color Distribution in the Picture "))
Color_Detector.launch(share=True,debug=True)