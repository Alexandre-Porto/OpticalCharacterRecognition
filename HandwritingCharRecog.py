import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

sys.path.append('../src')
from ocr.normalization import word_normalization, letter_normalization
from ocr import page, words, characters
from ocr.helpers import implt, resize
from ocr.tfhelpers import Model
from ocr.datahelpers import idx2char
import os
import cv2
import numpy as np
from cvlib import opencvfunction
from cvutil import cvdraw, cvcolor, cvperspective, cvtext
import math

LANG = 'en'
# You can use only one of these two
# You HABE TO train the CTC model by yourself using word_classifier_CTC.ipynb
MODEL_LOC_CHARS = '../models/char-clas/' + LANG + '/CharClassifier'
MODEL_LOC_CTC = '../models/word-clas/CTC/Classifier1'
CHARACTER_MODEL = Model(MODEL_LOC_CHARS)
CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')


class HandwritingCharRecog(opencvfunction.OpenCvGuiFunc):
    variables = opencvfunction.OpenCvGuiFunc.variables.copy()
    variables['model'] = opencvfunction.choice_variable(['base_model', 'ctc_model'], default='base_model')

    def __init__(self):
        super(HandwritingCharRecog,self).__init__()

    def recognise_char_sep_model(self, img):
        """Recognition using character model"""
        # Pre-processing the word
        img = word_normalization(
            img,
            60,
            border=False,
            tilt=True,
            hyst_norm=True)

        # Separate letters
        img = cv2.copyMakeBorder(
            img,
            0, 0, 30, 30,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        gaps = characters.segment(img, RNN=True)

        chars = []
        for i in range(len(gaps) - 1):
            char = img[:, gaps[i]:gaps[i + 1]]
            char, dim = letter_normalization(char, is_thresh=True, dim=True)
            # TODO Test different values
            if dim[0] > 4 and dim[1] > 4:
                chars.append(char.flatten())

        chars = np.array(chars)
        word = ''
        if len(chars) != 0:
            pred = CHARACTER_MODEL.run(chars)
            for c in pred:
                word += idx2char(c)

        return word

    def recognise_ctc(self, img):
        """Recognising words using CTC Model."""
        img = word_normalization(
            img,
            64,
            border=False,
            tilt=False,
            hyst_norm=False)
        length = img.shape[1]
        # Input has shape [batch_size, height, width, 1]
        input_imgs = np.zeros(
            (1, 64, length, 1), dtype=np.uint8)
        input_imgs[0][:, :length, 0] = img

        pred = CTC_MODEL.eval_feed({
            'inputs:0': input_imgs,
            'inputs_length:0': [length],
            'keep_prob:0': 1})[0]

        word = ''
        for i in pred:
            word += idx2char(i)
        return word


    def apply_with_source_info(self, im, source_info):
        
        lines = self.get_last_info('lines')

        for line in lines:
            if self.values['model'] == 'base_model':
                print(" ".join([self.recognise_char_sep_model(im[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))
            else:
                print(" ".join([self.recognise_ctc(im[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))


        return im

    def visual_feedback(self, im):
        return im


