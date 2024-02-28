# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
Use Graphs to classify
"""
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras 
from keras import layers
import keras_cv
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import scipy.signal as sig
import networkx as nx
import scipy.sparse as sp
from scipy.linalg import block_diag
 # %% Preparing Data
data_dir = os.getcwd()

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
subject_data = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'Subject{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
del subject_number,subject_numbers,mat_fname,mat_contents,data_dir

S1 = subject_data['S1'][:,:]

# %% 

def plvfcn(eegData):
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def compute_plv(subject_data):
    idx = ['L', 'R', 'Re']
    plv = {field: np.zeros((22, 22, 160)) for field in idx}
    for i, field in enumerate(idx):
        for j in range(160):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r, re = plv['L'], plv['R'], plv['Re']
    yl, yr, yre = np.zeros((160, 1)), np.ones((160, 1)), np.full((160, 1), 2)
    img = np.concatenate((l, r, re), axis=2)
    y = np.concatenate((yl, yr, yre), axis=1).reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)
    return img, y

plv, y = compute_plv(subject_data['S1'][:,:])

def create_graphs(plv, threshold=0.7):
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.DiGraph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs

graphs = create_graphs(plv, threshold=0.9)

adj = np.zeros([22, 22, len(graphs)])
fv = np.zeros([22, 3, len(graphs)])

for i, G in enumerate(graphs):
    degrees = np.array(list(G.degree()))
    fv[:, 0, i] = degrees[:, 1]
    clustering_coefficients = nx.clustering(G).values()
    fv[:, 1, i] = np.array(list(clustering_coefficients))
    fv[:, 2, i] = list(nx.degree_centrality(G).values())
    adj[:, :, i] = nx.to_numpy_array(G)

def split(S1, y, train_p, val_p, test_p):
    num_samples = S1.shape[-1]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    S1_shuffled = S1[..., indices]
    y_shuffled = y[indices, :]
    num_train_samples = int(train_p * num_samples)
    num_val_samples = int(val_p * num_samples)
    num_test_samples = int(test_p * num_samples)
    train_S1 = S1_shuffled[..., :num_train_samples]
    train_y = y_shuffled[:num_train_samples, :]
    val_S1 = S1_shuffled[..., num_train_samples:num_train_samples+num_val_samples]
    val_y = y_shuffled[num_train_samples:num_train_samples+num_val_samples, :]
    test_S1 = S1_shuffled[..., num_train_samples+num_val_samples:num_train_samples+num_val_samples+num_test_samples]
    test_y = y_shuffled[num_train_samples+num_val_samples:num_train_samples+num_val_samples+num_test_samples, :]
    return train_S1, train_y, val_S1, val_y, test_S1, test_y

train_adj, train_y, val_adj, val_y, test_adj, test_y = split(adj, y, 0.3, 0.2, 0.5)
train_fv, _, val_fv, _, test_fv, _ = split(fv, y, 0.3, 0.2, 0.5)

train_adj_list = [train_adj[:, :, i] for i in range(train_adj.shape[2])]
test_adj_list = [test_adj[:, :, i] for i in range(test_adj.shape[2])]
val_adj_list = [val_adj[:, :, i] for i in range(val_adj.shape[2])]

train_adj_s = block_diag(*train_adj_list)
test_adj_s = block_diag(*test_adj_list)
val_adj_s = block_diag(*val_adj_list)

train_fv = train_fv.transpose(2, 0, 1).reshape(-1, train_fv.shape[1])
val_fv = val_fv.transpose(2, 0, 1).reshape(-1, val_fv.shape[1])
test_fv = test_fv.transpose(2, 0, 1).reshape(-1, test_fv.shape[1])

#%%






