# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:17:55 2019

@author: isaac
"""
import tensorflow as tf
import os
import numpy as np

import commons.utils as utils
from commons.RNN_CTC_base import RNN_CTC_base

class BLSTM(RNN_CTC_base):
    def lstm_cell(self, n_hidden, dropout = 1):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True), 
                      input_keep_prob = dropout, output_keep_prob = dropout, state_keep_prob = dropout)

    def rnn_cell(self):        
        rnn_out, _, _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [self.lstm_cell(self.n_hidden, self.dropout_ph) for i in range(2)],
            [self.lstm_cell(self.n_hidden, self.dropout_ph) for i in range(2)],
            self.X,
            dtype=tf.float32,
            sequence_length=self.len_x)
        return rnn_out
    
    def logits_decode(self, logits, seq_length):
#        decoded, probs = tf.nn.ctc_beam_search_decoder(logits, seq_length, 100, 1)
#        return decoded[0],probs
        decoded, probs = tf.nn.ctc_greedy_decoder(logits, seq_length) #greedy is faster
        return decoded[0], probs
    
    def optimizer_fn(self, lr, loss):
        return tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
        
    def lr_scheduler(self, x,e,d):
        if e == 0:
            self.lr_penalty = 1
        
        if e != (self.n_epochs-1):
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)/((1-((e-1)/self.n_epochs)**8)) 
        else:
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)
            
        return x
    
    def __init__(self, n_feats, n_hidden, n_classes, logsdir = None, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(n_feats = n_feats, n_classes = n_classes, logsdir = logsdir, **kwargs)




def normalize_data(X_train, X_test, folder_name):
    train_matrix = []
    for x in X_train:
        train_matrix.extend(x)
    train_matrix = np.array(train_matrix)
    
    mean = np.array(train_matrix).mean(0)
    std = np.array(train_matrix).std(0)
        
    X_train = [(x-mean)/std for x in X_train]
    X_test = [(x-mean)/std for x in X_test]
    
    return np.array(X_train), np.array(X_test)


if __name__ == "__main__":
    
    folder_name = "/timit-feats/" # This folder contains X_train.npy as a list of np.arrays
    logdir = "./LSTM_MFCC_RBM/" # Output folder
    
    epochs = 30
    lr = 0.0015
    batchsize = 8
    dropout = 0.85
    
    n_feats = 500 # Esto da igual, se recalculan solas
    n_hidden = 128
    n_classes = 39 + 1
    
    X_train_path = folder_name + "X_train.npy"
    X_train = np.array(utils.load_npy(X_train_path))
    X_test_path = folder_name + "X_test.npy"
    X_test = np.array(utils.load_npy(X_test_path))
    
    y_train_path = folder_name + "y_train.npy"
    y_train = np.array(utils.load_npy(y_train_path))
    y_test_path = folder_name + "y_test.npy"
    y_test = np.array(utils.load_npy(y_test_path))

    X_train, X_test = normalize_data(X_train, X_test, folder_name)
    
    # Remove features with 0 variance (for RBM mostly)
    train_matrix = np.array(utils.to_matrix(X_train))
    std = train_matrix.std(0)
    good_features = np.where(std > 0)[0]
    X_train = np.array([x[:,good_features] for x in X_train])
    X_test = np.array([x[:,good_features] for x in X_test])

    print("Number of features: " + str(good_features))
    
    n_feats = X_train[0].shape[1]
        
    # for quick tests
#    X_train,y_train = utils.get_mini_dataset(X_train,y_train,500)
#    X_test, y_test = utils.get_mini_dataset(X_test, y_test,100)
    
    # Prepare y data
    y_train = utils.remove_q(utils.to_ctc(utils.reduce_labels(y_train, input_index = True, output_index = True)))
    y_test = utils.remove_q(utils.to_ctc(utils.reduce_labels(y_test, input_index = True, output_index = True)))
    
    rnn = BLSTM(n_feats = n_feats, n_hidden = n_hidden, n_classes = n_classes,
                logsdir = logdir)
    
    rnn.set_data(X_train, X_test, y_train, y_test)
    
    del(X_train, X_test, y_train, y_test) # To save memory
        
    rnn.fit(n_epochs=epochs,
            batch_size=batchsize,
            learning_rate=lr,
            dropout = dropout,
            shuffle=True,
            tboard = True,
            verbose=True,
            low_memory = True)

    rnn.save_weights(logdir)
