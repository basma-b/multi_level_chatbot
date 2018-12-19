# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import pickle
import keras, os
from keras.models import *
from keras.utils import np_utils
from keras.layers import *
from utilities import my_callbacks
import argparse
import keras
from utilities.data_helper import *


def main():
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--emb_trainable', type='bool', default=True, help='Whether fine tune embeddings')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--hidden_size_lstm', type=int, default=200, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--input_dir', type=str, default='./dataset/', help='Input dir')
    parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
    parser.add_argument('--model_fname', type=str, default='model/model.h5', help='Model filename')
    parser.add_argument('--embedding_file', type=str, default='embeddings/embeddings.vec', help='Embedding filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    np.random.seed(args.seed)
 
    print("Starting...")
    
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    
    print('Now indexing word vectors...')

    embeddings_index = {}
    f = open(args.embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()
    
    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_dir + 'params.pkl', 'rb'))
    

    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    print("Now loading embedding matrix...")
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words , args.emb_dim))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Now building dual encoder lstm model...")
    
    # define lstm encoder
    
    encoder = Sequential()
    encoder_input  = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding = Embedding(output_dim=args.emb_dim,
                            input_dim=MAX_NB_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=args.emb_trainable
                            )
    embedded_input = embedding(encoder_input)
    output = LSTM(units=args.hidden_size)(embedded_input)
    encoder = Model(encoder_input, [output, embedded_input])
    print(encoder.summary())
    
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # encode the context and the response
    context_branch, context_embed = encoder(context_input)
    response_branch, response_embed = encoder(response_input)
    
    # compute the sequence level similarity vector 
    S = keras.layers.multiply([context_branch, response_branch])
    
    # compute the word level similarity matrix
    embed_mul = keras.layers.dot([context_embed, response_embed], axes=2)
    # transform the word level similarity matrix into a vector
    W = LSTM(units=200)(embed_mul)
    
    # concatenate the word and sequence level similarity vectors
    concatenated = keras.layers.concatenate([S, W])

    out = Dense((1), activation = "sigmoid") (concatenated)

    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                    optimizer=args.optimizer)
    
    
    print(model.summary())
    
    print("Now loading data...")
    
    train_c, train_r, train_l = pickle.load(open(args.input_dir + 'train.pkl', 'rb'))
    test_c, test_r, test_l = pickle.load(open(args.input_dir + 'test.pkl', 'rb'))
    dev_c, dev_r, dev_l = pickle.load(open(args.input_dir + 'dev.pkl', 'rb'))
    
    print('Found %s training samples.' % len(train_c))
    print('Found %s dev samples.' % len(dev_c))
    print('Found %s test samples.' % len(test_c))
    
    print("Now training the model...")
    
    histories = my_callbacks.Histories()
    
    bestAcc = 0.0
    patience = 0 
    
    print("\tbatch_size={}, nb_epoch={}".format(args.batch_size, args.n_epochs))
    
    for ep in range(1, args.n_epochs):
                
        model.fit([train_c, train_r], train_l,
                batch_size=args.batch_size, epochs=1, callbacks=[histories],
                validation_data=([dev_c, dev_r], dev_l), verbose=1)

        curAcc =  histories.accs[0]
        if curAcc >= bestAcc:
            bestAcc = curAcc
            patience = 0
            
            if args.save_model:
                print("Now saving the model... at {}".format(args.model_fname))
                model.save(args.model_fname)
                
        else:
            patience = patience + 1
        
        # stop training the model when patience = 5
        if patience > 5:
            print("Early stopping at epoch: "+ str(ep))
            break

if __name__ == "__main__":
    main()