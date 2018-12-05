#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:20:18 2018

@author: raju
"""

import cv2
import math


class PossibleChar:

    # constructor
    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [X, Y, W, H] = self.boundingRect

        self.BoundingRectX = X
        self.BoundingRectY = Y
        self.BoundingRectWidth = W
        self.BoundingRectHeight = H

        self.BoundingRectArea = self.BoundingRectWidth * self.BoundingRectHeight

        self.CenterX = (self.BoundingRectX + self.BoundingRectX + self.BoundingRectWidth) / 2
        self.CenterY = (self.BoundingRectY + self.BoundingRectY + self.BoundingRectHeight) / 2

        self.DiagonalSize = math.sqrt((self.BoundingRectWidth ** 2) + (self.BoundingRectHeight ** 2))

        self.AspectRatio = float(self.BoundingRectWidth) / float(self.BoundingRectHeight)
