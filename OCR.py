# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:21:34 2019

@author: Porto
"""

def recognise(img):
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
    for i in range(len(gaps)-1):
        char = img[:, gaps[i]:gaps[i+1]]
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

def recognise_2(img):
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
   
def main():
    
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
    
    #%matplotlib inline
    plt.rcParams['figure.figsize'] = (15.0, 10.0)
    
    IMG = '../data/pages/test4.jpg'    # 1, 2, 3
    LANG = 'en'
    # You can use only one of these two
    # You HABE TO train the CTC model by yourself using word_classifier_CTC.ipynb
    MODEL_LOC_CHARS = '../models/char-clas/' + LANG + '/CharClassifier'
    MODEL_LOC_CTC = '../models/word-clas/CTC/Classifier1'
    
    CHARACTER_MODEL = Model(MODEL_LOC_CHARS)
    CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')
    
    image = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
    implt(image)
    
    # Crop image and get bounding boxes
    crop = page.detection(image)
    implt(crop)
    boxes = words.detection(crop)
    lines = words.sort_words(boxes)
    
    implt(crop)
    for line in lines:
        print(" ".join([recognise(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))
    
    implt(crop)
    for line in lines:
        print(" ".join([recognise_2(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line]))
        
main()