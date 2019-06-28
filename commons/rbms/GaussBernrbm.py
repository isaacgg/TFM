# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:11:30 2019

@author: isaac
"""
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from RBM_base import RBM_base

class GaussBerRBM(RBM_base):
    _id = "GaussBernRBM"

    def __init__(self, n_visible, n_hidden, k, logsdir, **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        super().__init__(n_visible, n_hidden, k, logsdir, **kwargs)
        
    def init_w(self):
        return tf.get_variable("w", 
                               shape=[self.n_visible, self.n_hidden],
                               trainable = False,
                               initializer = tf.initializers.random_normal(mean=0.0,stddev=0.001,))
        
    def init_visible_b(self):
        return tf.Variable(0*tf.ones([self.n_visibles]),
#                           trainable = False,
                           dtype=tf.float32, name = "visible_bias") #Hinton recomienda iniciarlos a  log[pi/(1−pi)]

    def init_hidden_b(self):
        return tf.Variable(0*tf.ones([self.n_hidden]), dtype=tf.float32) #Hinton recomienda iniciarlos a 0 o a -4


    def visible_activation(self, x, probs):
        return self.sample_gaussian(x, probs)
    
    def hidden_activation(self, x, probs):
        return self.sample_bernoulli(x, probs)

    def lr_scheduler(self, x,e,d):     
        if e == 0:
            self.lr_penalty = 1
       
        if e != (self.n_epochs-1):
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)/((1-((e-1)/self.n_epochs)**8)) 
        else:
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)
                        
        return x