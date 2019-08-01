# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:57:50 2019

@author: Porto
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

sys.path.append('../src')
from ocr.normalization import word_normalization

from ocr.normalization import letter_normalization
from ocr import page
from ocr import words
#from ocr import characters

from ocr.helpers import resize
#from ocr.tfhelpers import Model
from ocr.datahelpers import idx2char

import os
import cv2
import numpy as np
from cvlib import opencvfunction
from cvutil import cvdraw, cvcolor, cvperspective, cvtext
import math


class PageDetectionBreta(opencvfunction.OpenCvGuiFunc):
    variables = opencvfunction.OpenCvGuiFunc.variables.copy()

    def __init__(self):
        super(PageDetectionBreta,self).__init__()
     
    def apply_with_source_info(self, im, source_info):
        
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Crop image and get bounding boxes
        im = page.detection(image)
        
        return im

    def visual_feedback(self, im):
        return im
