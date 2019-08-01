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
from ocr.tfhelpers import Model
from ocr.datahelpers import idx2char

import os
import cv2
import numpy as np
from cvlib import opencvfunction
from cvutil import cvdraw, cvcolor, cvperspective, cvtext
import math

from cvutil import cvdraw

LANG = 'en'
'''
# You can use only one of these two
# You HABE TO train the CTC model by yourself using word_classifier_CTC.ipynb
MODEL_LOC_CHARS = '../models/char-clas/' + LANG + '/CharClassifier'
MODEL_LOC_CTC = '../models/word-clas/CTC/Classifier1'
CHARACTER_MODEL = Model(MODEL_LOC_CHARS)
CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')
'''
class WordDetectionBreta(opencvfunction.OpenCvGuiFunc):
    variables = opencvfunction.OpenCvGuiFunc.variables.copy()
    variables['inverted colors feedback'] = opencvfunction.checkbox_variable(False)
    
    def __init__(self):
        super(WordDetectionBreta,self).__init__()
        

    def apply_with_source_info(self, im, source_info):
        
        boxes = words.detection(im)
        lines = words.sort_words(boxes)
        
        
        self.add_info('bbxy', boxes)       
        self.add_info('lines', lines)
        
        return im

    def visual_feedback(self, im):
        plt.rcParams['figure.figsize'] = (15.0, 10.0)
        
        
        # Crop image and get bounding boxes
        boxes = words.detection(im)
        
        if self.values['inverted colors feedback']:
            im[:,:,:] = 255-im[:,:,:]
        
        for box in boxes:
            cv2.rectangle(im,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
        
        '''
        for line in lines:
            if self.values['model'] == 'base_model':
                print(" ".join([self.recognise_char_sep_model(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))
            else:
                print(" ".join([self.recognise_ctc(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))
        '''

        
        return im


