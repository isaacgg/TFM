# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:24:49 2019

@author: isaac
"""
import sys
import os
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from RBM_base import RBM_base

class GaussReLURBM(RBM_base):
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
        with tf.name_scope("visible_activation"):
            return self.sample_gaussian(x, probs)
    
    def hidden_activation(self, x, probs):
        with tf.name_scope("hidden_activation"):
            return self.sample_relu(x, probs)
    
    def get_features(self, x):
        return self._session.run(self.h_act_clean, {self.X: x})
    
    def get_features2(self, x):
        return self._session.run(self.h_prob, {self.X: x})
    
    def gibbs_step(self, h):
        with tf.name_scope("gibbs_step"):
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
        
        positive_grad = self.calc_positive_gradient(self.h_act_clean) #Hinton recomienda usar aquí las probabilidades en todo,                                                                   
        negative_grad = self.calc_negative_gradient(v_recon_mf, h_recon_act_clean)
        
        self.d_w = self.momentum_fn(self.d_w, self.lr*(self.calc_delta_w(positive_grad, negative_grad) - self.l1_ph*tf.abs(self.w)))
        self.d_vb = self.momentum_fn(self.d_vb, self.lr*self.calc_delta_visible_bias(v_recon_mf))#/128#esto estaba con v_recon_mf_bias
        self.d_hb = self.momentum_fn(self.d_hb, self.lr*self.calc_delta_hidden_bias(self.h_act_clean, h_recon_act_clean))#/128#Esto estaba con self.h_probs

#        positive_grad = self.calc_positive_gradient(self.h_act) #Hinton recomienda usar aquí las probabilidades en todo,                                                                   
#        negative_grad = self.calc_negative_gradient(v_recon_mf, h_recon_act)
#        
#        self.d_w = self.momentum_fn(self.d_w, self.lr*(self.calc_delta_w(positive_grad, negative_grad) - self.l1_ph*tf.abs(self.w)))
#        self.d_vb = self.momentum_fn(self.d_vb, self.lr*self.calc_delta_visible_bias(v_recon_mf))#/128#esto estaba con v_recon_mf_bias
#        self.d_hb = self.momentum_fn(self.d_hb, self.lr*self.calc_delta_hidden_bias(self.h_act, h_recon_act))#/128#Esto estaba con self.h_probs


        update_fn = [self.w.assign_add(self.d_w), self.hidden_bias.assign_add(self.d_hb), self.visible_bias.assign_add(self.d_vb)]
        return update_fn    
    
    def lr_scheduler(self, x,e,d):
        if e == 0:
            self.lr_penalty = 1
            self.error_count = 0
            
#        if e==self.n_epochs//2:
#            self.lr_penalty = 1/2;
#            
#        if e==3*self.n_epochs//4:
#            self.lr_penalty = 1/4;
#            
#        if e==6*self.n_epochs//4:
#            self.lr_penalty = 1/8;

#        if e>12:
            
#        if np.abs(d) < 0.001:
#            self.error_count += 1
#        else:
#            self.error_count = 0
#        
#        if (self.lr_penalty < 0.05) and self.error_count > 10:
#            self.lr_penalty *= 2
            
#        if (d < 0) and (self.lr_penalty > 0.01):
#            self.lr_penalty *= 1/2
            
#        else:
#            if self.lr_penalty < 1:
#                self.lr_penalty *= 1.1
        
        
        
        if e != (self.n_epochs-1):
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)/((1-((e-1)/self.n_epochs)**8)) 
        else:
            x *= self.lr_penalty*(1-(e/self.n_epochs)**8)
                        
        return x