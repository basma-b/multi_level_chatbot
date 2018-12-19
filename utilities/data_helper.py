from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np
import sys
import glob, os, csv, re
from collections import Counter
import itertools
from keras.preprocessing import sequence
from keras import backend as K

def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx())

def compute_recall_ks_no_label(probas):
    recall_k = {}
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print ('group_size: %d' % group_size)
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall_no_label(probas, k, group_size)
                print ('recall@%d' % k, recall_k[group_size][k])
    return recall_k

def recall_no_label(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)

def compute_recall_ks(probas, labels):
    recall_k = {}
    group_size = 10
    recall_k[group_size] = {}
    print ('group_size: %d' % group_size)
    for k in [1, 2, 5]:
        if k < group_size:
            recall_k[group_size][k] = recall(probas, labels, k, group_size)
            print ('recall@%d' % k, recall_k[group_size][k]*100.0)
    return recall_k

def compute_precision_ks(probas, labels):
    group_size = 10
    precision = precision_1(probas, labels, group_size)
    print ('precision@1', precision*100.0)
    return precision

def precision_1(probas, labels, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    k = 1
    for i in range(n_batches):
        
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        tab_batch = labels[i*test_size:(i+1)*test_size][:group_size]
        indexes = [index for index in range(len(tab_batch)) if tab_batch[index] == 1]
        idx = indices[0]
        if idx in indexes:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)


def recall(probas, labels, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        tab_batch = labels[i*test_size:(i+1)*test_size][:group_size]
        indexes = [index for index in range(len(tab_batch)) if tab_batch[index] == 1]
        cpt = 0
        for idx in indexes:
            if idx in indices:
                cpt += 1
        n_correct += float(cpt)/len(indexes)
    return float(n_correct) / (len(probas) / test_size)


def mrr(out, th):
  num_queries = len(out) / th
  MRR = 0.0
  for qid in range(num_queries):
    candidates = out[qid*th:(qid+1)*th]
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == 1:
        MRR += float(1.0) / (i + 1)
        break
  return float(MRR * 100.0) / num_queries

def map(out, th):
  num_queries = len(out) / th
  MAP = 0.0
  for qid in range(num_queries):
    candidates = out[qid*th:(qid+1)*th]
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == 1:
        num_correct += 1
        precisions.append(float(num_correct)/(i+1))
    
    if precisions:
      avg_prec = float(sum(precisions))/len(precisions)
    MAP += avg_prec
  return float(MAP *100.0)/ num_queries


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
