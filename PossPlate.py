import cv2
import numpy as np

class PossPlate:

    def __init__(self):
        self.frmPlate = None
        self.frmGrayscale = None
        self.frmThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""


