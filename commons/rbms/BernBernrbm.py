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
    def __init__(self, n_visible, n_hidden, k, logsdir, **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        super().__init__(n_visible, n_hidden, k, logsdir, **kwargs)

    def init_w(self):
        return tf.get_variable("w", shape=[self.n_visible, self.n_hidden],
                               trainable=False,
           initializer=tf.contrib.layers.xavier_initializer())

    def visible_activation(self, x, probs):
        return self.sample_bernoulli(x, probs)
    
    def hidden_activation(self, x, probs):
        return self.sample_bernoulli(x,probs)
        #        return self.sample_relu(x, probs)

#    def sample_relu(self, x, sigma):
#        if sigma == 0:
#            return tf.nn.relu(x)
#        return tf.nn.relu(tf.random_normal(shape = tf.shape(x), mean = x, stddev = sigma/3))

    def reconstruct(self):
        h_mf = tf.matmul(self.X, self.w) + self.hidden_bias
        h_prob = tf.nn.sigmoid(h_mf)
        h_act = self.hidden_activation(h_mf, 0)
        
        v_mf = tf.matmul(h_act, tf.transpose(self.w)) + self.visible_bias
        v_act = self.visible_activation(v_mf, 0)
        return v_act
    
    def get_features(self, x):
        return self._session.run(self.h_act, {self.X: x})
    
    def gibbs_step(self, h):
        v_recon_mf_w = tf.matmul(h, tf.transpose(self.w)) + self.visible_bias #Duda, para las unidades visibles no especifica que usar
        v_recon_prob_w = tf.nn.sigmoid(v_recon_mf_w)
        v_recon_act_w = self.visible_activation(v_recon_mf_w, v_recon_prob_w)
        h_recon_mf_w = tf.matmul(v_recon_act_w, self.w) + self.hidden_bias
        h_recon_prob_w = tf.nn.sigmoid(h_recon_mf_w)
        h_recon_act_w = self.hidden_activation(h_recon_mf_w, h_recon_prob_w)
        
        return v_recon_act_w, v_recon_prob_w, v_recon_mf_w, h_recon_act_w, h_recon_prob_w, h_recon_mf_w
    
    def optimizer_fn(self):
        v_recon_act, v_recon_prob, v_recon_mf, h_recon_act, h_recon_prob, h_recon_mf = self.contrastive_divergence(self.h_act)
        h_recon_act_clean = self.hidden_activation(h_recon_mf, 0)
        
        positive_grad = self.calc_positive_gradient(self.h_prob) #Hinton recomienda usar aquí las probabilidades en todo,                                                                   
        negative_grad = self.calc_negative_gradient(v_recon_prob, h_recon_prob)
        
        self.d_w = self.momentum_fn(self.d_w, self.lr*(self.calc_delta_w(positive_grad, negative_grad) - self.l1_ph*tf.abs(self.w)))
        self.d_vb = self.momentum_fn(self.d_vb, self.lr*self.calc_delta_visible_bias(v_recon_prob)) #esto estaba con v_recon_mf_bias
        self.d_hb = self.momentum_fn(self.d_hb, self.lr*self.calc_delta_hidden_bias(self.h_prob, h_recon_prob))#Esto estaba con self.h_probs

        update_fn = [self.w.assign_add(self.d_w), self.hidden_bias.assign_add(self.d_hb), self.visible_bias.assign_add(self.d_vb)]
        return update_fn    
    
    def lr_scheduler(self, x,e,d):
        if e == 0:
            self.lr_penalty = 1
#        if e>25:
#            if d < 0:
#                self.lr_penalty *= 1/2
#            else:
#                if self.lr_penalty < 1:
#                    self.lr_penalty *= 1.1
        
        if e != (self.n_epochs-1):
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)/((1-((e-1)/self.n_epochs)**8)) 
        else:
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)
                        
        return x