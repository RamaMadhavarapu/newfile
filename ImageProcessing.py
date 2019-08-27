import cv2
import numpy as np
import math

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(imOrg):
    frmGrayscale = extractValue(imOrg)
    frmMaxContrastGrayscale = maximizeContrast(frmGrayscale)
    height, width = frmGrayscale.shape
    frmBlurred = np.zeros((height, width, 1), np.uint8)
    frmBlurred = cv2.GaussianBlur(frmMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    frmThresh = cv2.adaptiveThreshold(frmBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return frmGrayscale, frmThresh

def extractValue(imOrg):
    height, width, numChannels = imOrg.shape
    frmHSV = np.zeros((height, width, 3), np.uint8)
    frmHSV = cv2.cvtColor(imOrg, cv2.COLOR_BGR2HSV)
    frmHue, frmSaturation, frmValue = cv2.split(frmHSV)
    return frmValue

def maximizeContrast(frmGrayscale):
    height, width = frmGrayscale.shape
    frmTopHat = np.zeros((height, width, 1), np.uint8)
    frmBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frmTopHat = cv2.morphologyEx(frmGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    frmBlackHat = cv2.morphologyEx(frmGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
    frmGrayscalePlusTopHat = cv2.add(frmGrayscale, frmTopHat)
    frmGrayscalePlusTopHatMinusBlackHat = cv2.subtract(frmGrayscalePlusTopHat, frmBlackHat)

    return frmGrayscalePlusTopHatMinusBlackHat










