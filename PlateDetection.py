import cv2
import numpy as np
import math
import MainProgram
import random

import ImageProcessing
import CharDetection
import PossPlate
import PossChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(frmOriginalScene):
    listOfPossiblePlates = []       
    height, width, numChannels = frmOriginalScene.shape
    frmGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    frmThreshScene = np.zeros((height, width, 1), np.uint8)
    frmContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if MainProgram.showSteps == True:
        cv2.imshow("0", frmOriginalScene)
        frmGrayscaleScene, frmThreshScene = ImageProcessing.preprocess(frmOriginalScene)         

    if MainProgram.showSteps == True: 
        cv2.imshow("1a", frmGrayscaleScene)
        cv2.imshow("1b", frmThreshScene)
    listOfPossibleCharsInScene = findPossibleCharsInScene(frmThreshScene)

    if MainProgram.showSteps == True: 
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        frmContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        cv2.drawContours(frmContours, contours, -1, MainProgram.SLR_WHITE)
        cv2.imshow("2b", frmContours)
    listOfListsOfMatchingCharsInScene = CharDetection.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    if MainProgram.showSteps == True:
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        frmContours = np.zeros((height, width, 3), np.uint8)
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            cv2.drawContours(frmContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        cv2.imshow("3", frmContours)
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                  
            possiblePlate = extractPlate(frmOriginalScene, listOfMatchingChars)        

        if possiblePlate.frmPlate is not None:                         
            listOfPossiblePlates.append(possiblePlate)                 
    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")  # 13 with MCLRNF1 image

    if MainProgram.showSteps == True: 
        print("\n")
        cv2.imshow("4a", frmContours)
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)
            cv2.line(frmContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), MainProgram.SLR_RED, 2)
            cv2.line(frmContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), MainProgram.SLR_RED, 2)
            cv2.line(frmContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), MainProgram.SLR_RED, 2)
            cv2.line(frmContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), MainProgram.SLR_RED, 2)

            cv2.imshow("4a", frmContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].frmPlate)
            cv2.waitKey(0)

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
    return listOfPossiblePlates
def findPossibleCharsInScene(frmThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    frmThreshCopy = frmThresh.copy()

    frmContours, contours, npaHierarchy = cv2.findContours(frmThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = frmThresh.shape
    frmContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        if MainProgram.showSteps == True:
            cv2.drawContours(frmContours, contours, i, MainProgram.SLR_WHITE)
        possibleChar = PossChar.PossibleChar(contours[i])

        if CharDetection.checkIfPossibleChar(possibleChar):                   
            intCountOfPossibleChars = intCountOfPossibleChars + 1          
            listOfPossibleChars.append(possibleChar)                        
    if MainProgram.showSteps == True: 
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 with MCLRNF1 image
        cv2.imshow("2a", frmContours)

    return listOfPossibleChars
def extractPlate(frmOriginal, listOfMatchingChars):
    possiblePlate = PossPlate.PossiblePlate()           

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)    

    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    possPlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    height, width, numChannels = frmOriginal.shape     
    frmRotated = cv2.warpAffine(frmOriginal, rotationMatrix, (width, height))       # rotate the entire image

    frmCropped = cv2.getRectSubPix(frmRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.frmPlate = frmCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate












