#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:22:07 2018

@author: raju
"""

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables 
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(img):
    listOfPossiblePlates = []                 

    height, width, numChannels = img.shape

    imgGray = np.zeros((height, width, 1), np.uint8)
    imgThresh = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: 
        cv2.imshow("0", img)
   

    imgGray, imgThresh = Preprocess.preprocess(img)         # preprocess to get grayscale and threshold images

    if Main.showSteps == True: 
        cv2.imshow("Gray SCale Image", imgGray)
        cv2.imshow("Image threshold", imgThresh)
    

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThresh)

    if Main.showSteps == True: #
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene))) 

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    

     # given a list of all possible chars, find groups of matching chars
     # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True:
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str( len(listOfListsOfMatchingCharsInScene)))  

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            RandomBlue = random.randint(0, 255)
            RandomGreen = random.randint(0, 255)
            RandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
          

            cv2.drawContours(imgContours, contours, -1, (RandomBlue, RandomGreen, RandomRed))
       
        cv2.imshow("Image Contours", imgContours)
   

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                  
        possiblePlate = extractPlate(img, listOfMatchingChars)         

        if possiblePlate.imgPlate is not None:                        
            listOfPossiblePlates.append(possiblePlate)                  
   

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found") 

    if Main.showSteps == True: 
        print("\n")
        cv2.imshow("Image Contours", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("Image Contpurs", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("List of Possible plates", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)


        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []               
    CountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                      

        if Main.showSteps == True: 
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
       

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                  
            CountOfPossibleChars = CountOfPossibleChars + 1      
            listOfPossibleChars.append(possibleChar)                        
   

    if Main.showSteps == True: 
        print("\nstep 2 - len(contours) = " + str(len(contours)))  
        print("step 2 - CountOfPossibleChars = " + str(CountOfPossibleChars))  
        cv2.imshow("2a", imgContours)
   
    return listOfPossibleChars



def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           
    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.CenterX)        

    # calculate the center point of the plate
    PlateCenterX = (listOfMatchingChars[0].CenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].CenterX) / 2.0
    PlateCenterY = (listOfMatchingChars[0].CenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].CenterY) / 2.0

    ptPlateCenter = PlateCenterX, PlateCenterY

    # calculate plate width and height
    PlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].BoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].BoundingRectWidth - listOfMatchingChars[0].BoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    TotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        TotalOfCharHeights = TotalOfCharHeights + matchingChar.BoundingRectHeight
    # end for

    AverageCharHeight = TotalOfCharHeights / len(listOfMatchingChars)

    PlateHeight = int(AverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    Opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].CenterY - listOfMatchingChars[0].CenterY
    Hypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    CorrectionAngleInRad = math.asin(Opposite / Hypotenuse)
    CorrectionAngleInDeg = CorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (PlateWidth, PlateHeight), CorrectionAngleInDeg )

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), CorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape     

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))      

    imgCropped = cv2.getRectSubPix(imgRotated, (PlateWidth, PlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         
    return possiblePlate

