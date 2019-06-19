# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:57:53 2018

@author: isaac
"""

import tensorflow as tf
import numpy as np
import sys

import time

from DefaultTfModel import DefaultTfModel
class RNN_base(DefaultTfModel):
    
    def __init__(self, n_feats = 26, n_hidden = 94, n_classes = 61, momentum = 0, use_ctc = True, logs_dir = None, lr_scheduler = lambda x,e,d:x):
        super().__init__()
                
        self.logs_dir = logs_dir
        
        self.n_classes = n_classes
        self.n_feats = n_feats
        self.n_hidden = n_hidden
        self.momentum = momentum
        self.use_ctc = use_ctc
                
        self._build_model()

        if self.logs_dir is not None:
            self.train_writer = tf.summary.FileWriter(logs_dir+"/logs/train", graph=self._graph)
            self.test_writer = tf.summary.FileWriter(logs_dir+"/logs/test", graph=self._graph)

        self._session = tf.Session(graph = self._graph)
        self._session.run(self.init)
        
        self.lr_scheduler = lr_scheduler
    
    def dense_to_sparse(self, dense_tensor, sequence_length):
        indices = tf.where(tf.sequence_mask(sequence_length))
        values = tf.gather_nd(dense_tensor, indices)
        shape = tf.shape(dense_tensor, out_type=tf.int64)
        return tf.SparseTensor(indices, values, shape)
    
     
    def _build_model(self):
        with self._graph.as_default():            
            self.lr = tf.placeholder(name = 'lr', shape=(), dtype = tf.float32)
            self.dropout = tf.placeholder(name = 'placeholder', shape=(), dtype = tf.float32)

            with tf.name_scope("variables"):
                w = tf.get_variable(name = "W", shape = [self.n_hidden*2, self.n_classes], dtype = tf.float32, initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype = tf.float32))
                b = tf.get_variable(name = 'b', shape=[self.n_classes], initializer = tf.constant_initializer(0))

            with tf.name_scope("Inputs"):
                self.X = tf.placeholder(name = 'X', shape=[None, None, self.n_feats], dtype = tf.float32) #[batchsize, seqlen, input_dims]
                self.len_x = tf.placeholder(name = 'lens_x', shape = [None], dtype = tf.int32)
                
                if self.use_ctc:
                    self.yIdx = tf.placeholder(tf.int64)
                    self.yVals = tf.placeholder(tf.int32)
                    self.yShape = tf.placeholder(tf.int64)
                    self.y = tf.SparseTensor(self.yIdx, self.yVals, self.yShape)

                else:
                    self.y = tf.placeholder(name = 'y', shape=[None, None], dtype = tf.int32) #[batchsize, seqlen]
                    self.len_y = tf.placeholder(name = 'lens_y', shape = [None], dtype = tf.int32)
                    
                    maxlen = tf.shape(self.y)[1]

                batchsize = tf.shape(self.X)[0]

            with tf.name_scope("rnn"):
                rnn_out = self.forward_pass() #[batch_size, seqlen, n_hidden]

            if self.use_ctc:
                with tf.name_scope("softmask"):
                    shape = tf.shape(self.X) #batchsize, seq_len, nfeats
                    batchsize, max_timesteps = shape[0], shape[1]

                    rnn_out_flat = tf.reshape(rnn_out, [-1, self.rnn_units*2])#[batchsize*seqlen, n_hidden]

#                    rnn_out_flat = tf.reshape(rnn_out, [-1, self.n_hidden*2])

                    logits = tf.matmul(rnn_out_flat, w) + b #[batchsize*seqlen, n_classes]

                    logits = tf.reshape(logits,[batchsize, -1, self.n_classes]) #[batchsize, seqlen, n_classes]
                    logits = tf.transpose(logits,[1, 0, 2]) #[seqlen, batchsize, n_classes] <- ctc_greedy need it like this

                    self.decoded, self.probs = tf.nn.ctc_greedy_decoder(logits, self.len_x)
                            #ctc_beam_search_decoder

                    self.output = tf.sparse_tensor_to_dense(self.decoded[0])

                with tf.name_scope("calc_losses"):

                    sparse_y = self.y
                    
                    self.loss = tf.nn.ctc_loss(labels= sparse_y, inputs=logits, sequence_length=self.len_x, time_major = True)#, ctc_merge_repeated = True, time_major = True)
                    self.loss = tf.reduce_mean(self.loss)
                         
                with tf.name_scope("calc_accuracies"):
                    self.accuracy = tf.constant(1.0) - tf.reduce_mean(tf.edit_distance(tf.to_int32(self.decoded[0]), sparse_y))
                
            else:
                onehot_y = tf.one_hot(self.y, self.n_classes, name="onehot_y") #[batchsize, seqlen, n_classes]
                mask_y = tf.sequence_mask(self.len_y, maxlen=maxlen)
                
                with tf.name_scope("softmask"):
                    rnn_out_flat = tf.reshape(rnn_out, [-1, self.rnn_units*2])#[batchsize*seqlen, n_hidden]
                                        
                    logits = tf.matmul(rnn_out_flat, w) + b #[batchsize*seqlen, n_classes]
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

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam-op').minimize(self.loss)

            with tf.name_scope("summaries"):
                hist_b = tf.summary.histogram('b', b)
                hist_w = tf.summary.histogram("w", w)
                                        
                self.hists_vars = tf.summary.merge([hist_w] + [hist_b])
                
                self.loss_ph = tf.placeholder(tf.float32, shape = (), name="loss_placeholder")
                self.summary_loss = tf.summary.scalar("loss", self.loss_ph)
                
                self.acc_ph = tf.placeholder(tf.float32, shape = (), name="accuracy_placeholder")
                self.summary_acc = tf.summary.scalar("accuracy", self.acc_ph)
                self.hist_metrics = tf.summary.merge([self.summary_loss] + [self.summary_acc])
                
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p].tolist(), b[p].tolist()
    
    def decode(self, batch_x, len_x):
        return self._session.run([self.output, self.probs], feed_dict={self.X: batch_x, self.len_x:len_x})
        
    def  list_to_sparse_tensorlist_to (targetList): #delete this :(
        indices = []
        vals = []

        for tIdx, target in enumerate(targetList):
            for seqIdx, val in enumerate(target):
                indices.append([tIdx, seqIdx])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
        return (np.array(indices), np.array(vals), np.array(shape))
    
    def to_sparse(self, batch_y, seq_len):
        indices = np.vstack([np.vstack((np.repeat(idx, ly), np.arange(ly))).T for idx,ly in enumerate(seq_len)])
        shape = batch_y.shape
        mask = ~(np.ones(shape).cumsum(axis=1).T > seq_len).T
        values = batch_y[mask]
        return (indices, values, shape)
    
    def partial_fit(self, batch_x, len_x, batch_y, len_y):
        if self.use_ctc:
            _,acc,loss = self._session.run([self.optimizer, self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                                 self.y: self.to_sparse(batch_y, len_y),
                                                                                                 self.len_x: len_x,
                                                                                                 self.lr: self.learning_rate})       
        else:    
            _,acc,loss = self._session.run([self.optimizer, self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                                 self.y: batch_y,
                                                                                                 self.len_x: len_x,
                                                                                                 self.len_y: len_y,
                                                                                                 self.lr: self.learning_rate})
        return acc,loss
            
    def partial_test(self, batch_x, len_x, batch_y, len_y):
        if self.use_ctc:
            acc, loss = self._session.run([self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                 self.y: self.to_sparse(batch_y, len_y),
                                                                                 self.len_x: len_x})
            
        else:
            acc, loss = self._session.run([self.accuracy, self.loss], feed_dict={self.X: batch_x,
                                                                                 self.y: batch_y,
                                                                                 self.len_x: len_x,
                                                                                 self.len_y: len_y})
        return acc, loss

    def write_train_metrics(self, acc,loss, e):
        metrics = self._session.run(self.hist_metrics, feed_dict={self.acc_ph: acc, self.loss_ph: loss})
        self.train_writer.add_summary(metrics, e)
    
    def write_test_metrics(self, acc,loss, e):
        metrics = self._session.run(self.hist_metrics, feed_dict={self.acc_ph: acc, self.loss_ph: loss})
        self.test_writer.add_summary(metrics, e)
    
    def set_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def fit(self,
            X_train = None,
            y_train = None,
            X_test = None,
            y_test = None,
            n_epoches=10,
            batch_size=10,
            learning_rate=0.01,
            shuffle=False,
            tboard = True,
            verbose=True):
        assert n_epoches > 0
        
        self.learning_rate = learning_rate
        
        if all(v is not None for v in [X_train, X_test, y_train, y_test]):
            self.set_data(X_train, X_test, y_train, y_test)
            del(X_train, X_test, y_train, y_test)
        
        n_data_train = self.X_train.shape[0]
        n_data_test = self.X_test.shape[0]

        if batch_size > 0:
            n_batches_train = n_data_train // batch_size + (0 if n_data_train % batch_size == 0 else 1)
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_train = 1
            n_batches_test = 1

        if shuffle:
            self.X_train, self.y_train = self.unison_shuffled_copies(self.X_train, self.y_train)
            self.X_test, self.y_test = self.unison_shuffled_copies(self.X_test, self.y_test)
            inds_train = np.arange(n_data_train)
            inds_test = np.arange(n_data_test)
            
        len_x_train = np.array([len(x) for x in self.X_train])
        len_x_test = np.array([len(x) for x in self.X_test])
        len_y_train = np.array([len(y) for y in self.y_train])
        len_y_test = np.array([len(y) for y in self.y_test])

        X_train_cpy = np.zeros((len(self.X_train), np.max(len_x_train), self.n_feats))
        for i,j in enumerate(self.X_train):
            X_train_cpy[i,:len(j),:] = j
        del(self.X_train)
        
        X_test_cpy = np.zeros((len(self.X_test), np.max(len_x_test), self.n_feats))
        for i,j in enumerate(self.X_test):
            X_test_cpy[i,:len(j),:] = j
        del(self.X_test)
        
        y_train_cpy = np.zeros((len(self.y_train), np.max(len_y_train)))
        for i,j in enumerate(self.y_train):
            y_train_cpy[i,:len(j)] = j
        del(self.y_train)
            
        y_test_cpy = np.zeros((len(self.y_test), np.max(len_y_test)))
        for i,j in enumerate(self.y_test):
            y_test_cpy[i,:len(j)] = j
        del(self.y_test)
        del(j)

#        errs = []
#        accs = []
#        loss = []
        p_err_mean = np.inf
        for e in range(n_epoches):
            print("lr: " + str(self.learning_rate))
            
            if verbose:
                print('Epoch: {:d}'.format(e))

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
            
            start = time.time()
            
            for b in range(n_batches_train):
                batch_x = X_train_cpy[b * batch_size:(b + 1) * batch_size]
                batch_y = y_train_cpy[b * batch_size:(b + 1) * batch_size]
                batch_len_x = len_x_train[b * batch_size:(b + 1) * batch_size]
                batch_len_y = len_y_train[b * batch_size:(b + 1) * batch_size]
                acc,loss = self.partial_fit(batch_x, batch_len_x, batch_y, batch_len_y)
                if (loss == np.inf) or (acc == np.nan):
                    raise Exception("Naaaaaaaaaaaan train")                
                
                epoch_accs_train[epoch_accs_train_ptr] = acc
                epoch_loss_train[epoch_loss_train_ptr] = loss
                epoch_accs_train_ptr += 1
                epoch_loss_train_ptr += 1
            
            end = time.time()
            print("training took: " + str(end - start))
            
            for b in range(n_batches_test):
                batch_x = X_test_cpy[b * batch_size:(b + 1) * batch_size]
                batch_y = y_test_cpy[b * batch_size:(b + 1) * batch_size]
                batch_len_x = len_x_test[b * batch_size:(b + 1) * batch_size]
                batch_len_y = len_y_test[b * batch_size:(b + 1) * batch_size]
                if (loss == np.inf) or (acc == np.nan):
                    raise Exception("Naaaaaaaaaaaan test")
                    
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
                if self.logs_dir is not None:
                    self.write_train_metrics(epoch_accs_train_mean,epoch_loss_train_mean, e)
                    self.write_test_metrics(epoch_accs_test_mean,epoch_loss_test_mean, e)

            if verbose:
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
            
            if self.logs_dir is not None:
                self.save_weights(self.logs_dir,"weights.h5")
            
        return
    
    def save_weights(self, filename, name):
        return self.saver.save(self._session, filename)
    
    def test_fun(self):
        if not self.use_ctc:
            return self._session.run(self.test, feed_dict={self.X: np.array([np.arange(self.n_feats*10).reshape([-1,self.n_feats]),10*np.arange(self.n_feats*10).reshape([-1,self.n_feats])+1]), 
                                                           self.len_x: [9,5],
                                                           self.y: [np.arange(10),10*np.arange(10)+1],
                                                           self.len_y: [9,5]})
        else:
            return self._session.run(self.test, feed_dict={self.X: np.array([np.arange(self.n_feats*10).reshape([-1,self.n_feats]),10*np.arange(self.n_feats*10).reshape([-1,self.n_feats])+1]), 
                                                           self.len_x: [9,5],
                                                           self.y: [np.arange(10),10*np.arange(10)+1],
                                                           self.len_y: [9,5]})

if __name__ == "__main__":
    import pickle
   
    with open('./data_checkpoints/data_ctc_no_window/X_train_mfcc.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle)) 
    with open('./data_checkpoints/data_ctc_no_window/X_test_mfcc.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle)) 
    with open('./data_checkpoints/data_ctc_no_window/y_train.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle)) 
    with open('./data_checkpoints/data_ctc_no_window/y_test.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))
        
    
    epochs = 10
    lr = 0.1
    def lr_scheduler(x,e,d, epochs = epochs, lr_init = lr):
        x *= (1-(e/epochs)**8) 
        if d<0:
            x /= 2
        return x

#
#    ix = 4610
#    l,p = rnn.decode([X_train[ix]], [len(X_train[ix])])
#    np.sum(l == y_train[ix])/len(y_train[ix])
#    
    
    rnn = RNN_base(n_feats = 13, n_hidden = 30, n_classes = 62, use_ctc = True, logs_dir = "RNN_test", lr_scheduler = lr_scheduler)
#    t = rnn.test_fun()
    rnn.fit(X_train,
            y_train,
            X_test,
            y_test,
            n_epoches=10,
            batch_size=128,
            learning_rate=0.1,
            shuffle=False,
            tboard = True,
            verbose=True)