# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:38:27 2019

@author: isaac
"""
import tensorflow as tf
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))

import utils
from TF_base import TF_base

class RNN_framewise_base(TF_base):    
    """ PARAMETRIZABLES """
    def rnn_cell(self):
        def lstm_cell(lstm_size):
            return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(lstm_size),
                                                 input_keep_prob = self.dropout_ph, output_keep_prob = self.dropout_ph, state_keep_prob = self.dropout_ph)
        rnn_out, _ = tf.nn.dynamic_rnn(lstm_cell(20), self.X, dtype=tf.float32, sequence_length=self.len_x)  
        rnn_out = tf.concat(rnn_out,2)
        
        return rnn_out            
    
    def logits_decode(self, logits, seq_length):
        #return ctc_beam_search_decoder(logits, seqlen, 100, 1) <-- could also be
        decoded, probs = tf.nn.ctc_greedy_decoder(logits, seq_length)
        return decoded[0], probs
    
    def optimizer_fn(self, lr, loss):
        return tf.train.AdamOptimizer(learning_rate=lr, name='Adam-op').minimize(loss)
        
    def lr_scheduler(self, x,e,d):
        return x
    
    def dropout_scheduler(self, x, e, d):
        return x
    
    def init_w(self):
         return tf.get_variable(name = "W", shape = [self.n_rnn_hidden, self.n_classes],
                                    dtype = tf.float32, 
                                    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype = tf.float32))
    def init_b(self):
        return tf.get_variable(name = 'b', shape=[self.n_classes], initializer = tf.constant_initializer(0))
    
    def __init__(self, n_feats, n_classes, logsdir, **kwargs):
        super().__init__(logsdir)
        
        self.n_feats = n_feats
        self.n_classes = n_classes
        
        self.build_ctc()
        self.init_graph()
        self.init_writers()
        
        self.open_session()
        self._session.run(self.init_variables)
        
    def build_ctc(self):
        with self._graph.as_default():            
            self.lr = tf.placeholder(name = 'lr', shape=(), dtype = tf.float32)
            self.dropout_ph = tf.placeholder(name = 'placeholder', shape=(), dtype = tf.float32)
            
            with tf.name_scope("Inputs"):
                self.X = tf.placeholder(name = 'X', shape=[None, None, self.n_feats], dtype = tf.float32) #[batchsize, seqlen, input_dims]
                self.len_x = tf.placeholder(name = 'lens_x', shape = [None], dtype = tf.int32)
                
                self.y = tf.placeholder(name = 'y', shape=[None, None], dtype = tf.int32) #[batchsize, seqlen]
                self.len_y = tf.placeholder(name = 'lens_y', shape = [None], dtype = tf.int32)
                
                maxlen = tf.shape(self.y)[1]

                batchsize = tf.shape(self.X)[0]    
                
            with tf.name_scope("rnn"):
                rnn_out = self.rnn_cell() #[batch_size, seqlen, n_hidden]
                self.n_rnn_hidden = rnn_out.get_shape()[2] 
                
            with tf.name_scope("variables"):
                w = self.init_w()
                b = self.init_b()
                               
            with tf.name_scope("softmask"):
                onehot_y = tf.one_hot(self.y, self.n_classes, name="onehot_y") #[batchsize, seqlen, n_classes]
                mask_y = tf.sequence_mask(self.len_y, maxlen=maxlen)

#                rnn_out = tf.reshape(rnn_out, [-1, tf.shape(rnn_out)[2]]) #[batchsize*seqlen, n_hidden]
#
#                logits = tf.matmul(rnn_out, w) + b #[batchsize*seqlen, n_classes]
#
#                logits = tf.reshape(logits,[batchsize, -1, self.n_classes]) #[batchsize, seqlen, n_classes]
#                logits = tf.transpose(logits,[1, 0, 2]) #[seqlen, batchsize, n_classes] <- ctc_greedy likes it like this
#                
#                self.framewise_probs = tf.nn.softmax(logits, axis = 2)
#                self.framewise_output = tf.argmax(self.framewise_probs, axis = 2)
#
#                self.decoded, self.probs = self.logits_decode(logits, self.len_x)

#                self.ctc_output = tf.sparse_tensor_to_dense(self.decoded)
                
                
                rnn_out = tf.reshape(rnn_out, [-1, tf.shape(rnn_out)[2]])#[batchsize*seqlen, n_hidden]
                
                logits = tf.matmul(rnn_out, w) + b #[batchsize*seqlen, n_classes]
#                logits = tf.reshape(logits,[batchsize, -1, self.n_classes]) #[batchsize, seqlen, n_classes]
#                logits = tf.transpose(logits,[1, 0, 2]) #[seqlen, batchsize, n_classes] <- ctc_greedy likes it like this
                #First one sequence until mask is false, then the other sequence until mask is false                    
                self.probs = tf.nn.softmax(logits) #[batchsize*seqlen, n_classes]
                
                self.decoded = tf.argmax(self.probs, axis = 1) #[batchsize*seqlen]  like [0,23,54,12,...]
                self.decoded = tf.reshape(self.decoded, [batchsize, -1]) #[batchsize, seqlen]
                self.output = self.decoded
                
            with tf.name_scope("calc_losses"):
                y_oh_true = tf.boolean_mask(onehot_y, mask_y)
                logits_masked = tf.reshape(logits,[batchsize, -1, self.n_classes])
                logits_masked = tf.boolean_mask(logits_masked, mask_y)

                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels= y_oh_true, logits=logits_masked), name='loss')
                         
            with tf.name_scope("calc_accuracies"):
                y_true = tf.cast(tf.boolean_mask(self.y, mask_y), tf.int32)
                y_predicted = tf.cast(tf.boolean_mask(self.decoded, mask_y, axis = 0), tf.int32)
                
                correct_prediction = tf.equal(y_predicted, y_true, name='correct_pred')
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
                
            with tf.name_scope("optimizer"):
                self.optimizer = self.optimizer_fn(lr=self.lr, loss=self.loss)
                
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

            with tf.name_scope("vars_summaries"):
                hist_b = tf.summary.histogram('b', b)
                hist_w = tf.summary.histogram("w", w)
                                        
                self.hists_vars = tf.summary.merge([hist_w] + [hist_b])
                
            with tf.name_scope("extras"):
                self.saver = tf.train.Saver()
                self.init_variables = tf.global_variables_initializer()
                
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
#        return a[p].tolist(), b[p].tolist()

    def partial_fit(self, batch_x, len_x, batch_y, len_y):
        _,acc,loss = self._session.run([self.optimizer, self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                                 self.y: batch_y,
                                                                                                 self.len_y: len_y,
                                                                                                 self.len_x: len_x,
                                                                                                 
                                                                                                 self.lr: self.learning_rate,
                                                                                                 self.dropout_ph: self.dropout})       
        return acc,loss
            
    def partial_test(self, batch_x, len_x, batch_y, len_y):
        acc, loss = self._session.run([self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                 self.y: batch_y,
                                                                                 self.len_y: len_y,
                                                                                 self.len_x: len_x,
                                                                                 self.dropout_ph: 1})            
        return acc, loss
    
    def set_data(self, X_train, X_test, y_train, y_test): #SET THIS BEFORE USE fit() TO SAVE MEMORY
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        
    def fit(self,
        n_epochs=10,
        batch_size=10, #batch_size = 0 means use the whole data
        learning_rate=0.01,
        dropout = 1,
        shuffle=False,
        tboard = True,
        verbose=True,
        low_memory = True):
        
        assert n_epochs > 0
        assert all(v is not None for v in [self.X_train, self.X_test, self.y_train, self.y_test]), "Set the data first using set_data(X_train, X_test, y_train, y_test)"
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.dropout = dropout
        
        if verbose:
            print("Preparing data...")
        
        n_data_train = self.X_train.shape[0]
        n_data_test = self.X_test.shape[0]
    
        if batch_size > 0:
            n_batches_train = n_data_train // batch_size + (0 if n_data_train % batch_size == 0 else 1)
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_train = 1
            n_batches_test = 1
    
#        if shuffle: #ESTO NO ES NECESARIO PORQUE YA HAGO EL SHUFFLE EN CADA EPOCA
#            self.X_train, self.y_train = self.unison_shuffled_copies(self.X_train, self.y_train)
#            self.X_test, self.y_test = self.unison_shuffled_copies(self.X_test, self.y_test)
#            inds_train = np.arange(n_data_train)
#            inds_test = np.arange(n_data_test)
        
        if shuffle:
            inds_train = np.arange(n_data_train)
            inds_test = np.arange(n_data_test)
            
        len_x_train = np.array([len(x) for x in self.X_train])
        len_x_test = np.array([len(x) for x in self.X_test])
        len_y_train = np.array([len(y) for y in self.y_train])
        len_y_test = np.array([len(y) for y in self.y_test])
    
        """ ZERO PADDING """
        X_train_cpy = np.zeros((len(self.X_train), np.max(len_x_train), self.n_feats))
        for i,j in enumerate(self.X_train):
            X_train_cpy[i,:len(j),:] = j
        if low_memory: del(self.X_train)
        
        X_test_cpy = np.zeros((len(self.X_test), np.max(len_x_test), self.n_feats))
        for i,j in enumerate(self.X_test):
            X_test_cpy[i,:len(j),:] = j
        if low_memory: del(self.X_test)
        
        y_train_cpy = np.zeros((len(self.y_train), np.max(len_y_train)))
        for i,j in enumerate(self.y_train):
            y_train_cpy[i,:len(j)] = j
        if low_memory: del(self.y_train)
            
        y_test_cpy = np.zeros((len(self.y_test), np.max(len_y_test)))
        for i,j in enumerate(self.y_test):
            y_test_cpy[i,:len(j)] = j
        if low_memory: del(self.y_test)
        
        del(i,j)
    
        if verbose:
            print("Start training")
            
        p_err_mean = np.inf
        for e in range(n_epochs):        
            if verbose:
                print('Epoch: {:d}'.format(e), " --  Learning rate:", self.learning_rate)
    
            epoch_accs_train = np.zeros((n_batches_train,))
            epoch_accs_train_ptr = 0
            epoch_loss_train = np.zeros((n_batches_train,))
            epoch_loss_train_ptr = 0
            
            epoch_accs_test = np.zeros((n_batches_test,))
            epoch_accs_test_ptr = 0
            epoch_loss_test = np.zeros((n_batches_test,))
            epoch_loss_test_ptr = 0
    
            if shuffle:
                np.random.shuffle(inds_train)
                X_train_cpy = X_train_cpy[inds_train]
                y_train_cpy = y_train_cpy[inds_train]
                len_x_train = len_x_train[inds_train]
                len_y_train = len_y_train[inds_train]
                
                np.random.shuffle(inds_test)
                X_test_cpy = X_test_cpy[inds_test]
                y_test_cpy = y_test_cpy[inds_test]
                len_x_test = len_x_test[inds_test]
                len_y_test = len_y_test[inds_test]
            
            if verbose:
                start = time.time()
            
            for b in range(n_batches_train):
                batch_x = X_train_cpy[b * batch_size:(b + 1) * batch_size]
                batch_y = y_train_cpy[b * batch_size:(b + 1) * batch_size]
                batch_len_x = len_x_train[b * batch_size:(b + 1) * batch_size]
                batch_len_y = len_y_train[b * batch_size:(b + 1) * batch_size]
                acc,loss = self.partial_fit(batch_x, batch_len_x, batch_y, batch_len_y)
                if (loss == np.inf) or (loss == np.nan) or (acc == np.nan):
                    raise Exception("Nan in train")                
                
                epoch_accs_train[epoch_accs_train_ptr] = acc
                epoch_loss_train[epoch_loss_train_ptr] = loss
                epoch_accs_train_ptr += 1
                epoch_loss_train_ptr += 1
            
            if verbose:
                end = time.time()
                print("training time: " + str((end - start)/60), "minutes")
                start = time.time()
    
            
            for b in range(n_batches_test):
                batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
                batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]
                batch_len_x = len_x_test[b * batch_size:(b + 1) * batch_size]
                batch_len_y = len_y_test[b * batch_size:(b + 1) * batch_size]
                if (loss == np.inf) or (loss == np.nan) or (acc == np.nan):
                    raise Exception("Nan in test")
                    
                acc, loss = self.partial_test(batch_x, batch_len_x, batch_y, batch_len_y)
                epoch_accs_test[epoch_accs_test_ptr] = acc
                epoch_loss_test[epoch_loss_test_ptr] = loss
                epoch_accs_test_ptr += 1
                epoch_loss_test_ptr += 1
                
            epoch_accs_train_mean = epoch_accs_train.mean()
            epoch_loss_train_mean = epoch_loss_train.mean()
            epoch_accs_test_mean = epoch_accs_test.mean()
            epoch_loss_test_mean = epoch_loss_test.mean()
    
            if tboard:
                if self.logsdir is not None:
                    self.write_train_metrics(epoch_accs_train_mean,epoch_loss_train_mean, e)
                    self.write_test_metrics(epoch_accs_test_mean,epoch_loss_test_mean, e)
    
            if verbose:
                end = time.time()
                print("test time: " + str((end - start)/60), "minutes")
    
                
                print('Train acc: {:.4f}'.format(epoch_accs_train_mean))
                print('Train loss: {:.4f}'.format(epoch_loss_train_mean))
                print('Test acc: {:.4f}'.format(epoch_accs_test_mean))
                print('Test loss: {:.4f}'.format(epoch_loss_test_mean))
                print('')
                
                sys.stdout.flush()
            
            #LR SCHEDULER
            delta_err = p_err_mean - epoch_loss_train_mean
            p_err_mean = epoch_loss_train_mean
            self.learning_rate = self.lr_scheduler(learning_rate, e, delta_err)
            self.dropout = self.dropout_scheduler(dropout, e, delta_err)
            
            if self.logsdir is not None:
                self.save_weights()
        
        return
    
if __name__ == "__main__":    
    epochs = 10
    lr = 0.01

    rnn = RNN_framewise_base(n_feats = 600, n_classes = 41, logsdir = "./RNN_test/")
    
    X_train = utils.load_npy('../../data/RBM_hann_v2/X_train.npy')
    X_test = utils.load_npy('../../data/RBM_hann_v2/X_test.npy')
    y_train = utils.load_npy('../../data/RBM_hann_v2/y_train.npy')
    y_test = utils.load_npy('../../data/RBM_hann_v2/y_test.npy')
    
    y_train = [utils.collapse_num_labels(y) for y in y_train]
    y_test = [utils.collapse_num_labels(y) for y in y_test]
    
    y_train = utils.to_ctc(y_train)
    y_test = utils.to_ctc(y_test)
            
    rnn.set_data(X_train,
            X_test,
            y_train,
            y_test)
    
    del(X_train, y_train, X_test, y_test)

    rnn.fit(n_epochs=epochs,
            batch_size=10,
            learning_rate=lr,
            dropout = 0.5,
            shuffle=False,
            tboard = True,
            verbose=True)