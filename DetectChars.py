#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:21:25 2018

@author: raju
"""

import os

import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar

# module level variables 

kNearest = cv2.ml.KNearest_create()

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    allContoursWithData = []               
    validContoursWithData = []             

    try:
        # read in training classifications
        Classifications = np.loadtxt("./Classifications.txt", np.float32)                  
    except:                                                                                 
        print("error, unable to open Classifications.txt, exiting program\n")  
        os.system("pause")
        return False                                                                       
    

    try:
        # read in training images
        FlattenedImages = np.loadtxt("./Flattened_images.txt", np.float32)                 
    except:                                                                                 
        print("error, unable to open flattened_images.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        
    
      # reshape numpy array to 1d, necessary to pass to call to train
    Classifications = Classifications.reshape((Classifications.size, 1))     
    
    # set default K to 2
    kNearest.setDefaultK(2)                                                             
    # train KNN object
    kNearest.train(FlattenedImages, cv2.ml.ROW_SAMPLE, Classifications)           
    
    # if training successful
    return True                            

def detectCharsInPlates(listOfPossiblePlates):
    PlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          
        return listOfPossiblePlates             

    # at this point we can be sure the list of possible plates has at least one plate

    for possiblePlate in listOfPossiblePlates:         

        possiblePlate.imgGray, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     

        if Main.showSteps == True: 
            cv2.imshow("Possible image plate", possiblePlate.imgPlate)
            cv2.imshow("Possible gray scale image plate", possiblePlate.imgGray)
            cv2.imshow("Possible plate image threshold", possiblePlate.imgThresh)
      
        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: 
            cv2.imshow("Possible image plate threshold after eliminating gray areas", possiblePlate.imgThresh)
       

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGray, possiblePlate.imgThresh)

        if Main.showSteps == True: 
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, numChannels), np.uint8)
             # clear the contours list
            del contours[:]                                        

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
          

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("Image Contours", imgContours)
        

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, numChannels), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                RandomBlue = random.randint(0, 255)
                RandomGreen = random.randint(0, 255)
                RandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
           
                cv2.drawContours(imgContours, contours, -1, (RandomBlue, RandomGreen, RandomRed))
          
            cv2.imshow("Image Contours 2", imgContours)
       
        # if no groups of matching chars were found in the plate
        if (len(listOfListsOfMatchingCharsInPlate) == 0):			

            if Main.showSteps == True: 
                print("chars found in plate number " + str(
                    PlateCounter) + " = (none), click on any image and press a key to continue . . .")
                PlateCounter = PlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            

            possiblePlate.strChars = ""
            continue						
        
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)): 
             # sort chars from left to right                         
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.CenterX)       
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
 

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, numChannels), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                RandomBlue = random.randint(0, 255)
                RandomGreen = random.randint(0, 255)
                RandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
         

                cv2.drawContours(imgContours, contours, -1, (RandomBlue, RandomGreen, RandomRed))
   
            cv2.imshow("Image Contours 3", imgContours)
        
        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        LenOfLongestListOfChars = 0
        IndexOfLongestListOfChars = 0

                # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > LenOfLongestListOfChars:
                LenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                IndexOfLongestListOfChars = i
           

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[IndexOfLongestListOfChars]

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, numChannels), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
          
            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("Image Conours 4", imgContours)
     

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: 
            print("chars found in plate number " + str(PlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            PlateCounter = lateCounter + 1
            cv2.waitKey(0)
     

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)

    return listOfPossiblePlates




def findPossibleCharsInPlate(imgGray, imgThresh):
    listOfPossibleChars = []                       
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    _, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                       
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              
            listOfPossibleChars.append(possibleChar)       
        
    return listOfPossibleChars




def checkIfPossibleChar(possibleChar):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.BoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.BoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.BoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.AspectRatio and possibleChar.AspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False



def findListOfListsOfMatchingChars(listOfPossibleChars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []               
    
    for possibleChar in listOfPossibleChars:                       
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)      

        listOfMatchingChars.append(possibleChar)            

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                     
        
        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call
        
        # for each list of matching chars found by recursive call
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars: 
            # add to our original list of lists of matching chars
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)           
        
        break      

    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []            

    for possibleMatchingChar in listOfChars:               
        if possibleMatchingChar == possibleChar:    
                                                   
            continue                               
                    
        DistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        AngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        ChangeInArea = float(abs(possibleMatchingChar.BoundingRectArea - possibleChar.BoundingRectArea)) / float(possibleChar.BoundingRectArea)

        ChangeInWidth = float(abs(possibleMatchingChar.BoundingRectWidth - possibleChar.BoundingRectWidth)) / float(possibleChar.BoundingRectWidth)
        ChangeInHeight = float(abs(possibleMatchingChar.BoundingRectHeight - possibleChar.BoundingRectHeight)) / float(possibleChar.BoundingRectHeight)

        # check if chars match
        if (DistanceBetweenChars < (possibleChar.DiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            AngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            ChangeInArea < MAX_CHANGE_IN_AREA and
            ChangeInWidth < MAX_CHANGE_IN_WIDTH and
            ChangeInHeight < MAX_CHANGE_IN_HEIGHT):
             # if the chars are a match, add the current char to list of matching chars
            listOfMatchingChars.append(possibleMatchingChar)       
       
    return listOfMatchingChars                

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    X = abs(firstChar.CenterX - secondChar.CenterX)
    Y = abs(firstChar.CenterY - secondChar.CenterY)

    return math.sqrt((X ** 2) + (Y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    Adj = float(abs(firstChar.CenterX - secondChar.CenterX))
    Opp = float(abs(firstChar.CenterY - secondChar.CenterY))
    # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
    if Adj != 0.0:
        # if adjacent is not zero, calculate angle
        AngleInRad = math.atan(Opp / Adj)      
    else:
         # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
        AngleInRad = 1.5708                         
    # calculate angle in degrees
    AngleInDeg = AngleInRad * (180.0 / math.pi)       

    return AngleInDeg

# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours, but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)              
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            # if current char and other char are not the same char . . .
            if currentChar != otherChar:
                 #if current char and other char have center points at almost the same location . . .
                                                                            #
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.DiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    # if current char is smaller than other char
                    if currentChar.BoundingRectArea < otherChar.BoundingRectArea:
                        # if current char was not already removed on a previous pass . . .
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved: 
                            # then remove current char
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         
                       
                    else:
                        # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            # if other char was not already removed on a previous pass . . .
                            # then remove other char
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)          
      

    return listOfMatchingCharsWithInnerCharRemoved


# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""              

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    # sort chars from left to right
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.CenterX)        
    
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                    
    for currentChar in listOfMatchingChars:                                         
        pt1 = (currentChar.BoundingRectX, currentChar.BoundingRectY)
        pt2 = ((currentChar.BoundingRectX + currentChar.BoundingRectWidth), (currentChar.BoundingRectY + currentChar.BoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           
                
        imgROI = imgThresh[currentChar.BoundingRectY : currentChar.BoundingRectY + currentChar.BoundingRectHeight,
                           currentChar.BoundingRectX : currentChar.BoundingRectX + currentChar.BoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))          
        ROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        

        ROIResized = np.float32(ROIResized)               
        retval, Results, neigh_resp, dists = kNearest.findNearest(ROIResized, k = 1)            

        strCurrentChar = str(chr(int(Results[0][0])))           

        strChars = strChars + strCurrentChar                      
  

    if Main.showSteps == True: 
        cv2.imshow("Image Thresh COlor", imgThreshColor)
   

    return strChars

