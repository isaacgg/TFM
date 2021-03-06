# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:22:54 2019

@author: isaac
"""
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from RBM_base import RBM_base

class BernBernRBM(RBM_base):
    _id = "BernBernRBM"
    
    def __init__(self, n_visible, n_hidden, k, logsdir, **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        super().__init__(n_visible, n_hidden, k, logsdir, **kwargs)

    def init_w(self):
        return tf.get_variable("w", 
                               shape=[self.n_visible, self.n_hidden],
                               trainable=False,
                               initializer=tf.contrib.layers.xavier_initializer())

    def visible_activation(self, x, probs):
        return self.sample_bernoulli(x, probs)
    
    def hidden_activation(self, x, probs):
        return self.sample_bernoulli(x,probs)
    
    def lr_scheduler(self, x,e,d):
        if e == 0:
            self.lr_penalty = 1

        if e != (self.n_epochs-1):
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)/((1-((e-1)/self.n_epochs)**8)) 
        else:
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)
                        
        return x