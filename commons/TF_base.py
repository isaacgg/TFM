# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:26:58 2019

@author: isaac
"""

import tensorflow as tf
import numpy as np

class TF_base:
    _session = None
    _graph = None
    saver = None
    logsdir = None
    
    def __init__(self, logsdir):
        self.logsdir = logsdir
        self._graph = tf.Graph()
        
    def init_graph(self):             
        with self._graph.as_default():
            with tf.name_scope("summaries_placehodlers"):
                self.loss_ph = tf.placeholder(tf.float32, shape = (), name="loss_placeholder")
                self.acc_ph = tf.placeholder(tf.float32, shape = (), name="accuracy_placeholder")
                                
                self.summary_loss = tf.summary.scalar("loss", self.loss_ph)                
                self.summary_acc = tf.summary.scalar("accuracy", self.acc_ph)

                self.hist_metrics = tf.summary.merge([self.summary_loss] + [self.summary_acc])

    def momentum_fn(self, x_old, x_new):
        return self.momentum_ph * x_old +\
               self.lr * x_new * (1 - self.momentum_ph)

    def init_writers(self):
        self.train_writer = tf.summary.FileWriter(self.logsdir+"/logs/train", graph=self._graph)
        self.test_writer = tf.summary.FileWriter(self.logsdir+"/logs/test")        

    def write_train_metrics(self, acc, loss, e):
        metrics = self._session.run(self.hist_metrics, feed_dict={self.acc_ph: acc, self.loss_ph: loss})
        self.train_writer.add_summary(metrics, e)
    
    def write_test_metrics(self, acc, loss, e):
        metrics = self._session.run(self.hist_metrics, feed_dict={self.acc_ph: acc, self.loss_ph: loss})
        self.test_writer.add_summary(metrics, e)
        
    def open_session(self, config = None):
        if self._session is None:
            print("Open session")
            self._session = tf.InteractiveSession(graph = self._graph, config = config)
            
    def close_session(self):
        if self._session is not None:
            self._session.close()
            self._session = None
            
    def save_weights(self, filename = None):
        if filename is None:
            filename = self.logsdir
        return self.saver.save(self._session, filename)
    
    def load_weights(self, filename = None):
        if filename is None:
            filename = self.logsdir
        self.saver.restore(self._session, filename)
        
    """ utils """
    def to_sparse(self, batch_y, seq_len):
        indices = np.vstack([np.vstack((np.repeat(idx, ly), np.arange(ly))).T for idx,ly in enumerate(seq_len)])
        shape = batch_y.shape
        mask = ~(np.ones(shape).cumsum(axis=1).T > seq_len).T
        values = batch_y[mask]
        return (indices, values, shape)
        
    