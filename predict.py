# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Merge, Activation, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge
from keras.models import Model
from utilities import my_callbacks
import argparse
from utilities.data_helper import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import os
from keras import backend as K

def main():
    
    print("Loading model ...!")   
    # load our best model
    model = load_model('model/model.h5')
    # load test data
    test_c, test_r, test_l = pickle.load(open('dataset/test.pkl', 'rb'))
    # predict test score
    print("Perform on test set ...!")   
    y_pred = dual_encoder.predict([test_c, test_r])
    #compute metrics
    pred = y_pred[:,0]
    compute_recall_ks(pred, test_l) # recall@k
    compute_precision_ks(pred, test_l) #P@1

    sorted_probas = []
    pred_labels = []
    for i in range(len(pred)/10):
        
        candidates = pred[i*10:(i+1)*10]
        candidate_labels = test_l[i*10:(i+1)*10]

        sorted_probas = list(np.array(candidates).argsort()[::-1])
        pred_labels += list(np.array(candidate_labels)[sorted_probas])

    print ("MAP = ", map(pred_labels, 10)) #MAP
    print ("MRR = ", mrr(pred_labels, 10)) #MRR
    
if __name__ == "__main__":
    main()