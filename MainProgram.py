import cv2
import numpy as np
import os

import CharDetection
import PlateDetection
import PossPlate

SLR_BLACK = (0.0, 0.0, 0.0)
SLR_WHITE = (255.0, 255.0, 255.0)
SLR_YELLOW = (0.0, 255.0, 255.0)
SLR_GREEN = (0.0, 255.0, 0.0)
SLR_RED = (0.0, 0.0, 255.0)
showSteps = False
def main():
    blnKNNTrainingSuccessful = CharDetection.loadKNNDataAndTrainKNN()        
    if blnKNNTrainingSuccessful == False:                              
        print("\nerror: KNN traning was not successful\n") 
        return                                                          
    frmOriginalScene  = cv2.imread("ExampleImages/1.png")  #example            
    if frmOriginalScene is None:                          
        print("\nerror: image not read from file \n\n") 
        os.system("pause")                                  
        return
    listOfPossiblePlates = PlateDetection.detectPlatesInScene(frmOriginalScene)           
    listOfPossiblePlates = CharDetection.detectCharsInPlates(listOfPossiblePlates)       
    cv2.imshow("frmOriginalScene", frmOriginalScene)           
    if len(listOfPossiblePlates) == 0:                          
        print("\nno license plates were detected\n") 
    else:                                               
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        licPlate = listOfPossiblePlates[0]
        cv2.imshow("frmPlate", licPlate.frmPlate)   
        cv2.imshow("frmThresh", licPlate.frmThresh)
        if len(licPlate.strChars) == 0:             
            print("\nno characters were detected\n\n") 
            return                                         
        drawRedRectangleAroundPlate(frmOriginalScene, licPlate)         
        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  
        print("----------------------------------------")
        writeLicensePlateCharsOnImage(frmOriginalScene, licPlate)           
        cv2.imshow("frmOriginalScene", frmOriginalScene)               
        cv2.imwrite("frmOriginalScene.png", frmOriginalScene)         
    cv2.waitKey(0)
    return

def drawRedRectangleAroundPlate(frmOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)           
    cv2.line(frmOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SLR_RED, 2)       
    cv2.line(frmOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SLR_RED, 2)
    cv2.line(frmOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SLR_RED, 2)
    cv2.line(frmOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SLR_RED, 2)
def writeLicensePlateCharsOnImage(frmOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                            
    ptCenterOfTextAreaY = 0
    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0
    sceneHeight, sceneWidth, sceneNumChannels = frmOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.frmPlate.shape
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                    
    intFontThickness = int(round(fltFontScale * 1.5))   
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene
    intPlateCenterX = int(intPlateCenterX)              
    intPlateCenterY = int(intPlateCenterY)
    ptCenterOfTextAreaX = int(intPlateCenterX)        
    if intPlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          
    cv2.putText(frmOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SLR_YELLOW, intFontThickness)
if __name__ == "__main__":
    main()


















