#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:18:43 2018

@author: raju
"""


import cv2
import numpy as np



def preprocess(image):
    # convert image into gray scale
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    height, width = imgGray.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    
    # Applying TopHat and BlackHat morphoogy of kernal size (3,3)
    imgTopHat = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    imgBlackHat = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    # maximum contrast gray scale image
    imgGrayscalePlusTopHat = cv2.add(imgGray, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgGrayscalePlusTopHatMinusBlackHat, (5,5), 0)
    
    # applying gaussian thresholding filtering with black size 19 and weight 9
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    
    return imgGray, imgThresh

