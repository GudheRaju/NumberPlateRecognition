#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:22:34 2018

@author: raju
"""

import cv2
#import numpy as np
import os

import DetectChars
import DetectPlates
#import PossiblePlate

# module level variables 
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main():
    # attempt KNN training
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         
    # if KNN training was not successful
    if blnKNNTrainingSuccessful == False:                              
        print("\nerror: KNN traning was not successful\n")  
        return                                                         
    
    #imgOriginalScene  = cv2.imread("./test_image.jpg")              
    img  = cv2.imread("./plate.png")  

    if img is None:                           
        print("\n error: image not read from file \n\n")  
        os.system("pause")                                 
        return                                            
  
    # detect plates
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(img)           
    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        

    cv2.imshow("imgOriginalScene", img)           
    # if no plates were found
    if len(listOfPossiblePlates) == 0:                          
        print("\n no license plates were detected\n")  
    else:                                                       
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

         # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)  
        # show crop of plate and threshold of plate        
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     
            print("\nno characters were detected\n\n") 
            return                                         

        drawRedRectangleAroundPlate(img, licPlate)             
        print("\n license plate read from image = " + licPlate.strChars + "\n")  
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(img, licPlate)           
        cv2.imshow("imgOriginalScene", img)               
        cv2.imwrite("imgOriginalScene.png", img)           
  

    cv2.waitKey(0)					

    return
def drawRedRectangleAroundPlate(img, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            
    cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)



def writeLicensePlateCharsOnImage(img, licPlate):
    ptCenterOfTextAreaX = 0                            
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                         

    sceneHeight, sceneWidth, sceneNumChannels = img.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    FontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    FontScale = float(plateHeight) / 30.0                    
    FontThickness = int(round(FontScale * 1.5))           
    textSize, baseline = cv2.getTextSize(licPlate.strChars, FontFace, FontScale, FontThickness)       

    # unpack roatated rect into center point, width and height, and angle
    ( (PlateCenterX, PlateCenterY), (PlateWidth, PlateHeight), CorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    PlateCenterX = int(PlateCenterX)              
    PlateCenterY = int(PlateCenterY)

    ptCenterOfTextAreaX = int(PlateCenterX)      
    if PlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(PlateCenterY)) + int(round(plateHeight * 1.6))     
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(PlateCenterY)) - int(round(plateHeight * 1.6))      
 
    textSizeWidth, textSizeHeight = textSize              

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))        
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

    # write the text on the image
    cv2.putText(img, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), FontFace, FontScale, SCALAR_YELLOW, FontThickness)


if __name__ == "__main__":
    main()
