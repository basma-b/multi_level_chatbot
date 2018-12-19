from __future__ import division
import keras
import numpy as np
from utilities.data_helper import *


class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}): 
        self.accs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(([self.validation_data[0], self.validation_data[1]]))
        
        pred = y_pred[:,0]
        test_l = self.validation_data[2]
        
        recall_k = compute_recall_ks(pred, test_l)
        compute_precision_ks(pred, test_l)
        
        sorted_probas = []
        pred_labels = []
        for i in range(int(len(pred)/10)):
            
            candidates = pred[i*10:(i+1)*10]
            candidate_labels = test_l[i*10:(i+1)*10]

            sorted_probas = list(np.array(candidates).argsort()[::-1])
            pred_labels += list(np.array(candidate_labels)[sorted_probas])
    
        print ("MAP = ", map(pred_labels, 10))
        print ("MRR = ", mrr(pred_labels, 10))
        
        self.accs.append(mrr(pred_labels, 10))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
