# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:59:43 2019

@author: isaac
"""

import pickle
import numpy as np
import python_speech_features as psf
from scipy.io.wavfile import read as wavread
from plotly.offline import plot
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(__file__))

""" Labels and TIMIT related"""
from timit_utils import *

""" Signal """
def read_wav(wav_file):
    fr, wav = wavread(wav_file)
    wav = wav/np.max(np.abs(wav))
    return wav, fr

def frame_wav(wav, winlen, winstep, winfunc = lambda x:np.ones((x,))):
    return psf.sigproc.framesig(sig=wav, frame_len=winlen, frame_step=winstep, winfunc = winfunc)

def to_matrix(X):
    matrix = []
    for x in X:
        matrix.extend(x)
    return np.array(matrix)

"""Data"""
def load_npy(file):
    return np.load(file)

def save_npy(data, file):
    if type(file) == str: #ESTO ES POR SI SE PONEN LOS ARGUMENTOS AL REVÉS
        _file = file
    else:
        _file = data
        data = file
        
    np.save(arr = data, file = _file)

def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(file, data):
    if type(file) == str: #ESTO ES POR SI SE PONEN LOS ARGUMENTOS AL REVÉS
        _file = file
    else:
        _file = data
        data = file

    with open(_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)  

def get_mini_dataset(X, y, n):
    n_data = len(X)
    inds = np.arange(n_data)
    np.random.shuffle(inds)
    X_cpy = X[inds]
    y_cpy = y[inds]
    return X_cpy[:n], y_cpy[:n]

def add_deltas(X,n):
    _X = []
    for x in X:
        d = psf.delta(x, n)
        _X.append(np.concatenate((x,d), axis = 1))
    return np.array(_X)
    
""" Visualization """
def plot_scatter_matrix(df, category = "phon", labels = [], figsize =(16,16)):
    if labels != []:
        df = df[df[category].isin(labels)]

    labels = df[category].unique()
    
    
    fig,axes = plt.subplots(len(labels), len(labels), figsize = figsize, sharex=True, sharey=True)
    
    for ax, col in zip(axes[0], labels):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], labels):
        ax.set_ylabel(row, rotation=0, size='large')
    
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            df[category]==labels[col]
            axes[row,col].scatter(df[df[category]==labels[col]]["pca0"].values, df[df[category]==labels[col]]["pca1"].values, s = 0.5)
            axes[row,col].scatter(df[df[category]==labels[row]]["pca0"].values, df[df[category]==labels[row]]["pca1"].values, s = 0.5)
#            axes[row,col].set_xlim(0,10)
#            axes[row,col].set_ylim(0,10)
    plt.show()


def plot_weights(rbm, rows, size = (60,60)):
    cols = int(np.ceil(rbm.n_hidden/rows))       #12*12
    fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)
    plot_num = 0

    _ws = rbm.get_weights()[0]
    for i in range(rows):
        for j in range(cols):
            axes[i,j].plot(np.real(_ws[:,plot_num]))
            plot_num += 1
    plt.show()

def plot_input_bias(rbm):
    _ws = rbm.get_weights()[1]
    plt.stem(_ws)
    plt.show()

def plot_hidden_bias(rbm):    
    _ws = rbm.get_weights()[2]
    plt.stem(_ws)
    plt.show()

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                         'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                         'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                         'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                         'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def plot_3d(pca, labels):
    def get_spaced_colors(n):
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        
        return ['rgb(' + str(int(i[:2], 16))+ ',' + str(int(i[2:4], 16)) + ',' + str(int(i[4:], 16)) + ')' for i in colors]

    labels_unique = np.unique(labels)
    if len(labels_unique) <= 10:
        colorslist = DEFAULT_PLOTLY_COLORS
    else:
        N = len(labels_unique)        
        colorslist = get_spaced_colors(N)

        
    data = []
    for ix, p in enumerate(labels_unique):
        ixs = np.where(np.array(labels) == p)[0] #all frames with label p
        trace1 = go.Scatter3d(
            x=pca[ixs,0],
            y=pca[ixs,1],
            z=pca[ixs,2],
            mode='markers',
            marker=dict(
                size=1,
                line=dict(
                    color= colorslist[ix%len(colorslist)],
                    width=10
                ),
                opacity=1
            ),
            name = p
        )
        data.append(trace1)
        
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig)
    
""" Other """
def suspender():
    from ctypes import windll
    # Suspender.
    if not windll.powrprof.SetSuspendState(False, False, False):
        print("No se ha podido suspender el sistema.")
        