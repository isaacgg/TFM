# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:06:25 2019

@author: isaac
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(__file__))

import utils
from TF_base import TF_base

class RBM_base(TF_base):    
    """ UTILS """
    def get_weights(self):
        return self._session.run(self.w)
    
    def get_input_bias(self):
        return self._session.run(self.visible_bias)

    def get_hidden_bias(self):
        return self._session.run(self.hidden_bias)

    def plot_weights(self, rows, size = (60,60)):
        cols = int(np.ceil(self.n_hidden/rows))       #12*12
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)
        plot_num = 0
    
        _ws = self.get_weights()
        for i in range(rows):
            for j in range(cols):
                axes[i,j].plot(np.real(_ws[:,plot_num]))
                plot_num += 1
        plt.show()
    
    def plot_input_bias(self):
        plt.figure()
        _ws = self.get_input_bias()
        plt.stem(_ws)
        plt.show()
    
    def plot_hidden_bias(self):
        plt.figure()
        _ws = self.get_hidden_bias()
        plt.stem(_ws)
        plt.show()
    
    """ OVERRIDE PARENT"""
    def write_train_metrics(self, loss, e):
        metrics = self._session.run([self.summary_loss, self.hists_vars], feed_dict={self.loss_ph: loss})
        self.train_writer.add_summary(metrics[0], e)
        self.train_writer.add_summary(metrics[1], e)
        
    def write_test_metrics(self, loss, e):
        metrics = self._session.run(self.summary_loss, feed_dict={self.loss_ph: loss})
        self.test_writer.add_summary(metrics, e)
        
    """ INITS """
    def init_w(self):
#        tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        return tf.get_variable(name = "W", shape = [self.n_visibles, self.n_hidden],
                                    trainable = False,
                                    dtype = tf.float32, 
                                    initializer = tf.initializers.random_normal(mean=0.0,stddev=0.001,))
    def init_hidden_b(self):
        return tf.Variable(0*tf.ones([self.n_hidden]), dtype=tf.float32) #Hinton recomienda iniciarlos a 0 o a -4
    
    def init_visible_b(self):
        return tf.Variable(0*tf.ones([self.n_visibles]), dtype=tf.float32) #Hinton recomienda iniciarlos a  log[pi/(1−pi)]
 

    """ ACTIVATION AND SAMPLE FUNCTIONS"""
    def sample_gaussian(self, x, sigma):
        with tf.name_scope("sample_gaussian"):
            return tf.random_normal(tf.shape(x), mean=x, stddev=sigma, dtype=tf.float32)

    def sample_relu(self, x, sigma):
        with tf.name_scope("sample_relu"):
            if sigma == 0:
                return tf.nn.relu(x)
            return tf.nn.relu(tf.random_normal(shape = tf.shape(x), mean = x, stddev = sigma))
        
    def sample_bernoulli(self, x, sigma):
        with tf.name_scope("sample_bernoulli"):
            on_probs = tf.nn.sigmoid(x)
            if sigma == 0:
                return on_probs
    #            return tf.nn.relu(tf.sign(on_probs - 0.5*tf.ones_like(on_probs)))
            return tf.nn.relu(tf.sign(on_probs - tf.random_uniform(tf.shape(on_probs))))


    """ PARAMETRIZABLES """
    def hidden_activation(self, x, probs):
        with tf.name_scope("hidden_activation"):
            return self.sample_bernoulli(x, probs)

    def visible_activation(self, h, probs):
        with tf.name_scope("visible_activation"):
            return self.sample_bernoulli(h, probs)
    
    def reconstruct(self):
        with tf.name_scope("reconstruct"):
            h_mf = tf.matmul(self.X, self.w) + self.hidden_bias
            h_act = self.hidden_activation(h_mf, 0)
            v_mf = tf.matmul(h_act, tf.transpose(self.w)) + self.visible_bias
            v_act = self.visible_activation(v_mf, 0)
            return v_act
    """ END PARAMETRIZABLE """
    
    """ TRAINING FUNCTS """
    def calc_positive_gradient(self, h):
        with tf.name_scope("positive_corr"):
            return tf.matmul(tf.transpose(self.X), h)/self.batchsize
    
    def calc_negative_gradient(self, v, h):
        with tf.name_scope("negative_corr"):
            return tf.matmul(tf.transpose(v), h)/self.batchsize
    
    def calc_delta_w(self, positive_grad, negative_grad):
        with tf.name_scope("delta_w"):
            return positive_grad - negative_grad
        
    def calc_delta_hidden_bias(self, h1, h2):
        with tf.name_scope("delta_hidden_bias"):
            return tf.reduce_mean(h1-h2, 0)
    
    def calc_delta_visible_bias(self, v):
        with tf.name_scope("delta_visible_bias"):
            return tf.reduce_mean(self.X-v, 0)
        
    def gibbs_step(self, h):
        with tf.name_scope("gibbs_step"):
            v_recon_mf_w = tf.matmul(h, tf.transpose(self.w)) + self.visible_bias
            v_recon_prob_w = tf.nn.sigmoid(v_recon_mf_w)
            v_recon_act_w = self.visible_activation(v_recon_mf_w, v_recon_prob_w)
            h_recon_mf_w = tf.matmul(v_recon_act_w, self.w) + self.hidden_bias
            h_recon_prob_w = tf.nn.sigmoid(h_recon_mf_w)
            h_recon_act_w = self.hidden_activation(h_recon_mf_w, h_recon_prob_w)
            
            return v_recon_act_w, v_recon_prob_w, v_recon_mf_w, h_recon_act_w, h_recon_prob_w, h_recon_mf_w

    def contrastive_divergence(self, h):
        v_act = tf.zeros_like(self.X)
        v_prob = tf.zeros_like(self.X)
        v_mf = tf.zeros_like(self.X)
        h_act = tf.identity(h)                 #SIEMPRE usar de primeras h_act
        h_prob = tf.identity(h)
        h_mf = tf.identity(h)
        
        """ For the last update of the hidden units, it is silly to use stochastic binary states because nothing
        depends on which state is chosen. So use the probability itself to avoid unnecessary sampling noise.
        When using CDn, only the final update of the hidden units should use the probability. """
        #No se si se refiere a usar las probabilidades para calcular las hidden o que las activaciones de las hidden sean probabilidades
        for i in range(self.k):
            with tf.control_dependencies([v_act, v_prob, v_mf, h_act, h_prob, h_mf]):
                v_act, v_prob, v_mf, h_act, h_prob, h_mf = self.gibbs_step(h_act)
        #Hinton recomienda usar las probabilidades en las unidades ocultas para el último paso (Asumiendo unidades binarias)    
        #Hinton recomienda usar las probabilidades aquí y en el gibbs_step para las unidades visibles
        return v_act, v_prob, v_mf, h_act, h_prob, h_mf
#        return v_act, v_prob, v_mf, h_act, h_prob, h_mf #Duda, uso h_prob o ReLU(h_mf)
    
    
    def optimizer_fn(self):
        with tf.name_scope("optimizer"):
            v_recon_act, v_recon_prob, v_recon_mf, h_recon_act, h_recon_prob, h_recon_mf = self.contrastive_divergence(self.h_act)
            h_recon_act_clean = self.hidden_activation(h_recon_mf, 0)
            
            positive_grad = self.calc_positive_gradient(self.h_act_clean) #Hinton recomienda usar aquí las probabilidades en todo,                                                                   
            negative_grad = self.calc_negative_gradient(v_recon_mf, h_recon_act_clean)
            
            self.d_w = self.momentum_fn(self.d_w, self.lr*(self.calc_delta_w(positive_grad, negative_grad) - self.l1_ph*tf.abs(self.w)))
            self.d_vb = self.momentum_fn(self.d_vb, self.lr*self.calc_delta_visible_bias(v_recon_mf))#/128#esto estaba con v_recon_mf_bias
            self.d_hb = self.momentum_fn(self.d_hb, self.lr*self.calc_delta_hidden_bias(self.h_act_clean, h_recon_act_clean))#/128#Esto estaba con self.h_probs
    
            update_fn = [self.w.assign_add(self.d_w), self.hidden_bias.assign_add(self.d_hb), self.visible_bias.assign_add(self.d_vb)]
            return update_fn
        
    """ EXTRAS """
    def calc_hidden_from_data(self):
        h_mf = tf.matmul(self.X, self.w) + self.hidden_bias
        h_prob = tf.nn.sigmoid(h_mf)
        h_act = self.hidden_activation(h_mf, h_prob)
        return h_act, h_prob, h_mf
    
    def get_features(self, x):
        return self._session.run(self.h_act_clean, {self.X: x})
    
    def get_reconstructed(self, x):
        return self._session.run(self.v_reconstructed, {self.X: x})
    
    def restart_hidden_fn(self):
        self._session.run(self.restart_hidden_bias)

    """ CALLBACKS """
    def lr_scheduler(self, x,e,d):
        return x
    
    def dropout_scheduler(self, x, e, d):
        return x
    
    def epoch_callback(self, e):
        pass
    
    """ MAIN METHODS """
    def __init__(self, n_visibles, n_hidden, k, logsdir, **kwargs):        
        self.n_visibles = n_visibles
        self.n_hidden = n_hidden
        self.k = k
                
        super().__init__(logsdir)
        self.build_rbm()
        self.init_graph()
        self.init_writers()
        
        self.open_session()
        self._session.run(self.init_variables)

        
    def build_rbm(self):
        with self._graph.as_default():            
            self.lr = tf.placeholder(name = 'lr_ph', shape=(), dtype = tf.float32)
            self.dropout_ph = tf.placeholder(name = 'dropout_ph', shape=(), dtype = tf.float32)
            self.l1_ph = tf.placeholder(name = 'l1_ph', shape=(), dtype = tf.float32)
            self.momentum_ph = tf.placeholder(name = 'momentum_ph', shape=(), dtype = tf.float32)
            
            with tf.name_scope("Inputs"):
                self.X = tf.placeholder(name = 'X', shape=[None, self.n_visibles], dtype = tf.float32) #[batchsize, input_dims]

                self.batchsize = tf.cast(tf.shape(self.X)[0], dtype = tf.float32)    

            with tf.name_scope("variables"):
                self.w = self.init_w()
                self.hidden_bias = self.init_hidden_b()
                self.visible_bias = self.init_visible_b()
                
                self.restart_hidden_bias = self.hidden_bias.assign(tf.zeros_like(self.hidden_bias))
                
                #Las deltas se inicializan siempre a cero
                self.d_w = tf.Variable(tf.zeros([self.n_visibles, self.n_hidden]), dtype=tf.float32)
                self.d_vb = tf.Variable(tf.zeros([self.n_visibles]), dtype=tf.float32)
                self.d_hb = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)
                
            with tf.name_scope("sample_input_hidden"):
                self.h_act, self.h_prob, self.h_mf = self.calc_hidden_from_data()
                self.h_act_clean = self.hidden_activation(self.h_mf, 0)
                
                self.features = self.hidden_activation(tf.matmul(self.X, self.w) + self.hidden_bias,0)
 
            with tf.name_scope("optimizer"):
                self.optimizer = self.optimizer_fn()
                               
            with tf.name_scope("loss"):
                self.v_reconstructed = self.reconstruct()
                self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.X - self.reconstruct())))  
                
                self.optimizer_backprop = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                
            with tf.name_scope("vars_summaries"):
                hist_hidden_bias = tf.summary.histogram('hidden_bias', self.hidden_bias)
                hist_visible_bias = tf.summary.histogram('visible_bias', self.visible_bias)
                hist_w = tf.summary.histogram("w", self.w)
                                        
                self.hists_vars = tf.summary.merge([hist_w] + [hist_hidden_bias] + [hist_visible_bias])
                
            with tf.name_scope("extras"):
                self.saver = tf.train.Saver()
                self.init_variables = tf.global_variables_initializer()
                        
    def partial_fit(self, batch_x):
        _, loss = self._session.run([self.optimizer, self.loss], feed_dict={self.X: batch_x,
                                                                            self.lr: self.learning_rate,
                                                                            self.dropout_ph: self.dropout,
                                                                            self.momentum_ph: self.momentum,
                                                                            self.l1_ph: self.l1})       
        return loss
            
    def partial_test(self, batch_x):
        loss = self._session.run(self.loss, feed_dict={self.X: batch_x,
                                                       self.dropout_ph: 1})            
        return loss
    
    def partial_fine_tune(self, batch_x):
        _, loss = self._session.run([self.optimizer_backprop, self.loss], feed_dict={self.X: batch_x,
                                                                    self.lr: self.learning_rate,
                                                                    self.dropout_ph: self.dropout,
                                                                    self.momentum_ph: self.momentum,
                                                                    self.l1_ph: self.l1})       
        return loss

    
    def fine_tune(self,
            X_train,
            X_test,
            n_epochs = 10,
            batch_size = 10, #batch_size = 0 means use the whole data
            learning_rate = 0.01,
            dropout = 1,
            l1 = 0,
            momentum = 0,
            shuffle = False,
            tboard = True,
            verbose = True):
        
        assert n_epochs > 0
#        assert all(v is not None for v in [self.X_train, self.X_test]), "Set the data first using set_data(X_train, X_test, y_train, y_test)"
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.l1 = l1
        self.momentum = momentum
 
        if verbose:
            print("Preparing data...")
        
        n_data_train = X_train.shape[0]
        n_data_test = X_test.shape[0]
    
        if batch_size > 0:
            n_batches_train = n_data_train // batch_size + (0 if n_data_train % batch_size == 0 else 1)
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_train = 1
            n_batches_test = 1
    
        if shuffle:
            inds_train = np.arange(n_data_train)
            inds_test = np.arange(n_data_test)
                    
        if verbose:
            print("Start training")
            
        p_err_mean = np.inf
        for e in range(n_epochs):        
            if verbose:
                print('Epoch: {:d}'.format(e), " --  Learning rate:", self.learning_rate)
            epoch_loss_train = np.zeros((n_batches_train,))
            epoch_loss_train_ptr = 0
            
            epoch_loss_test = np.zeros((n_batches_test,))
            epoch_loss_test_ptr = 0
    
            if shuffle:
                np.random.shuffle(inds_train)
                X_train = X_train[inds_train]

                np.random.shuffle(inds_test)
                X_test = X_test[inds_test]
            
            if verbose:
                start = time.time()
            
            for b in range(n_batches_train):
                batch_x = X_train[b * batch_size:(b + 1) * batch_size]
                loss = self.partial_fine_tune(batch_x)
                if (loss == np.inf) or (loss == np.nan):
                    raise Exception("Nan in train")                
                
                epoch_loss_train[epoch_loss_train_ptr] = loss
                epoch_loss_train_ptr += 1
            
            if verbose:
                end = time.time()
                print("training time: " + str((end - start)/60), "minutes")
                start = time.time()
            
            for b in range(n_batches_test):
                batch_x = X_test[b * batch_size:(b + 1) * batch_size]
                if (loss == np.inf) or (loss == np.nan):
                    raise Exception("Nan in test")
                    
                loss = self.partial_test(batch_x)
                epoch_loss_test[epoch_loss_test_ptr] = loss
                epoch_loss_test_ptr += 1
                
            epoch_loss_train_mean = epoch_loss_train.mean()
            epoch_loss_test_mean = epoch_loss_test.mean()
    
            if tboard:
                if self.logsdir is not None:
                    self.write_train_metrics(epoch_loss_train_mean, e)
                    self.write_test_metrics(epoch_loss_test_mean, e)
    
            if verbose:
                end = time.time()
                print("test time: " + str((end - start)/60), "minutes")
                print('Train loss: {:.4f}'.format(epoch_loss_train_mean))
                print('Test loss: {:.4f}'.format(epoch_loss_test_mean))
                print('')
                
                sys.stdout.flush()
            
            delta_err = p_err_mean - epoch_loss_train_mean
            p_err_mean = epoch_loss_train_mean
            self.learning_rate = self.lr_scheduler(learning_rate, e, delta_err)
            self.dropout = self.dropout_scheduler(dropout, e, delta_err)
            self.epoch_callback(e)
            
            if self.logsdir is not None:
                self.save_weights()
        
    def fit(self,
            X_train,
            X_test,
            n_epochs = 10,
            batch_size = 10, #batch_size = 0 means use the whole data
            learning_rate = 0.01,
            dropout = 1,
            l1 = 0,
            momentum = 0,
            shuffle = False,
            tboard = True,
            verbose = True):
        
        assert n_epochs > 0
#        assert all(v is not None for v in [self.X_train, self.X_test]), "Set the data first using set_data(X_train, X_test, y_train, y_test)"
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.l1 = l1
        self.momentum = momentum

        
        if verbose:
            print("Preparing data...")
        
        n_data_train = X_train.shape[0]
        n_data_test = X_test.shape[0]
    
        if batch_size > 0:
            n_batches_train = n_data_train // batch_size + (0 if n_data_train % batch_size == 0 else 1)
            n_batches_test = n_data_test // batch_size + (0 if n_data_test % batch_size == 0 else 1)
        else:
            n_batches_train = 1
            n_batches_test = 1
    
        if shuffle:
            inds_train = np.arange(n_data_train)
            inds_test = np.arange(n_data_test)
                    
        if verbose:
            print("Start training")
            
        p_err_mean = np.inf
        for e in range(n_epochs):        
            if verbose:
                print('Epoch: {:d}'.format(e), " --  Learning rate:", self.learning_rate)
            epoch_loss_train = np.zeros((n_batches_train,))
            epoch_loss_train_ptr = 0
            
            epoch_loss_test = np.zeros((n_batches_test,))
            epoch_loss_test_ptr = 0
    
            if shuffle:
                np.random.shuffle(inds_train)
                X_train = X_train[inds_train]

                np.random.shuffle(inds_test)
                X_test = X_test[inds_test]
            
            if verbose:
                start = time.time()
            
            for b in range(n_batches_train):
                batch_x = X_train[b * batch_size:(b + 1) * batch_size]
                loss = self.partial_fit(batch_x)
                if (loss == np.inf) or (loss == np.nan):
                    raise Exception("Nan in train")                
                
                epoch_loss_train[epoch_loss_train_ptr] = loss
                epoch_loss_train_ptr += 1
            
            if verbose:
                end = time.time()
                print("training time: " + str((end - start)/60), "minutes")
                start = time.time()
            
            for b in range(n_batches_test):
                batch_x = X_test[b * batch_size:(b + 1) * batch_size]
                if (loss == np.inf) or (loss == np.nan):
                    raise Exception("Nan in test")
                    
                loss = self.partial_test(batch_x)
                epoch_loss_test[epoch_loss_test_ptr] = loss
                epoch_loss_test_ptr += 1
                
            epoch_loss_train_mean = epoch_loss_train.mean()
            epoch_loss_test_mean = epoch_loss_test.mean()
    
            if tboard:
                if self.logsdir is not None:
                    self.write_train_metrics(epoch_loss_train_mean, e)
                    self.write_test_metrics(epoch_loss_test_mean, e)
    
            if verbose:
                end = time.time()
                print("test time: " + str((end - start)/60), "minutes")
                print('Train loss: {:.4f}'.format(epoch_loss_train_mean))
                print('Test loss: {:.4f}'.format(epoch_loss_test_mean))
                print('')
                
                sys.stdout.flush()
            
            delta_err = p_err_mean - epoch_loss_train_mean
            p_err_mean = epoch_loss_train_mean
            self.learning_rate = self.lr_scheduler(learning_rate, e, delta_err)
            self.dropout = self.dropout_scheduler(dropout, e, delta_err)
            self.epoch_callback(e)
            
            if self.logsdir is not None:
                self.save_weights()
        
        return
    
#momentum: Hinton recomienda empezar en 0.5 y luego subirlo a 0.9 y si es inestable *1/2
#regularization: Hinton recomienda usar L2 o L1
if __name__ == "__main__":
    import os
    from scipy import signal
    import DataInfoExtractor as die
    
    n_visibles = 400
    n_hidden = 150 #25
    logsdir = "D:/TrabajoFinDeMaster/greluRBM_400_150/"
    
    #hyperparameters
    n_epochs = 80
    k = 1
    batchsize = 128
    lr = 0.001
    dropout = 1
    
    winsize = n_visibles #100
    winstep = 160
    if True:
        if not os.path.isdir(logsdir):
            os.makedirs(logsdir)
        window = signal.hann(winsize) #signal.hann(winsize)#np.ones(winsize)
        
        ie = die.DatasetInfoExtractor(dataset_folder = "./Database/TIMIT", checkpoints_folder = "./data_checkpoints")
        
        train_matrix = ie.get_train_matrix_from_wavs(winsize, winstep)
#        test_matrix = ie.get_test_matrix_from_wavs(400, 160)

        mean = np.array(train_matrix).mean()#(axis = 0)
        std = np.array(train_matrix).std()#(axis = 0)
        
        utils.save_pickle(mean, logsdir + "mean.pkl")
        utils.save_pickle(std, logsdir + "std.pkl")

        train_matrix = np.array([window*(r-mean)/(std) for r in train_matrix])
#        test_matrix = np.array([window*(r-mean)/(std) for r in test_matrix])
        
    RBM_base(n_visibles, n_hidden, k, logsdir)
    RBM_base.fit(train_matrix,
                n_epochs = n_epochs,
                batch_size = batchsize, #batch_size = 0 means use the whole data
                learning_rate = lr,
                dropout = dropout,
                shuffle = False,
                tboard = True,
                verbose = True,
                low_memory = False)