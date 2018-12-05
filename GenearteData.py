#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:48:07 2018

@author: raju
"""

import sys
import numpy as np
import cv2
import os

# module level variables 
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    # read training image
    img = cv2.imread("./training_chars.png")          

    if img is None:
        # print error message to std out if image not found           
        print("error: image not read from file \n\n")        
        os.system("pause")                                 
        return                                              
   
    # convert BGR image to gray scale 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    
    # apply gaussian blur to the gray image     
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                     
    # filter image black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)                                  

    cv2.imshow("imgThresh", imgThresh)      
    
    # it is better to make a copy of the filtered image, to avoid modification to the original image while finding contours
    imgThreshCopy = imgThresh.copy()        
    
    # finding contours for he filtered image
    _, Contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       

                              
    FlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    Classifications = []       

   # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    ValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for Contour in Contours:
        # if contour is big enough to consider                          
        if cv2.contourArea(Contour) > MIN_CONTOUR_AREA:          
            [X, Y, W, H] = cv2.boundingRect(Contour)   
           # draw rectangle around each contour 
           
            cv2.rectangle(img, (X, Y), (X+W, Y+H), (0, 255, 0), 2)                

            imgROI = imgThresh[Y:Y+H, X:X+W]                                  
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))   

            cv2.imshow("imgROI", imgROI)                   
            cv2.imshow("imgROIResized", imgROIResized)      
            cv2.imshow("training_numbers.png", img)      

            Char = cv2.waitKey(0)                   

            if Char == 37:                  
                sys.exit()                      
            elif Char in ValidChars:     

                Classifications.append(Char)                                              
                FlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) 
                FlattenedImages = np.append(FlattenedImages, FlattenedImage, 0)
                
    # convert classifications list of ints to numpy array of floats
    Classifications = np.array(Classifications, np.float32)                  
    # flatten numpy array of floats to 1d so we can write to file later
    Classifications = Classifications.reshape((Classifications.size, 1))   

    print("\n\n TRAINING COMPLETED SUCCESSFULLY !!\n")

    np.savetxt("./Classifications.txt", Classifications)          
    np.savetxt("./Flattened_images.txt",FlattenedImages)         
    
  
    cv2.destroyAllWindows()           

    return


if __name__ == "__main__":
    main()
