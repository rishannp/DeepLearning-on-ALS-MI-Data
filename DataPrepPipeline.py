# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.

Building a dataset and testing a GAT model. 

The goal is to treat each instance seperately. So a giant for loop with a function for each step.This allows for unique sizing of each i.e. edge_index per datalist


To Do: 
- ICA for Brain Wave Isolation
- Optimising parameters


https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import os
from os.path import dirname, join as pjoin
import time
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from scipy.integrate import simps
from torch_geometric.data import Data
from tqdm import tqdm
#%%

# % Preparing Data
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

S1 = subject_data['S39'][:,:]

#%%

########### PLV Function ###############
# Computes Phase Locking Values using Hilbert Transform Method
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
#######################################

############# Create a Graph #################
def create_graphs(plv, threshold):
    G = nx.DiGraph()
    G.add_nodes_from(range(plv.shape[0]))
    for u in range(plv.shape[0]):
        for v in range(plv.shape[0]):
            if u != v and plv[u, v] > threshold:
                G.add_edge(u, v, weight=plv[u, v])
    return G

##############################################

############ Create Edge Index ###############
def edge_idx(adj,threshold):
    # Initialize lists to store source and target nodes
    source_nodes = []
    target_nodes = []
    
    # Iterate through each element of the adjacency matrix
    for row in range(adj.shape[0]):
        for col in range(adj.shape[1]):
            # Check if there's an edge
            if adj[row, col] >= threshold:
                # Add source and target nodes to the lists
                source_nodes.append(row)
                target_nodes.append(col)
            else:
                # If no edge exists, add placeholder zeros to maintain size
                source_nodes.append(0)
                target_nodes.append(0)
    
    # Create edge index as a LongTensor
    #edge_index = [source_nodes,target_nodes]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return edge_index
#####################################################

############## Band Pass EEG data #####################
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data
#######################################################

##### Adding 10, 3rd Dimensions to the Raw Data #######
def add_dims(data,dims):
    data = data[..., np.newaxis]
    data = np.copy(data) * np.ones(dims)   
    data = np.transpose(data,[1,0,2])
    return data
########################################################

###### Band Power Function ############################
def bandpower(data,low,high,fs):
    # Define window length (2s)
    win = 2* fs
    freqs, psd = signal.welch(data, fs, nperseg=win)
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)   
    # # Plot the power spectral density and fill the delta area
    # plt.figure(figsize=(7, 4))
    # plt.plot(freqs, psd, lw=2, color='k')
    # plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 40])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram") 
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    # Compute the absolute power by approximating the area under the curve
    power = simps(psd[idx_delta], dx=freq_res)
    return power
############################################################

####### Full Pre-processing function #######################
def preprocess(data,idx,band,fs):
    #Preallocations
    plv = {field: np.zeros((22, 22)) for field in idx}
    adj = {field: np.zeros((22, 22)) for field in idx}
    edge_index = {field: {} for field in idx}
    bpdata = {field: {} for field in idx}
    G = {field: {} for field in idx}
    x = {field: np.zeros((22, 10)) for field in idx}
    y = {field: np.zeros((1, 1)) for field in idx}
    data_list = []
    
    for j, field in enumerate(idx):
        plv[field][:, :] = plvfcn(data[field][:,:])
        G[field][j] = create_graphs(plv[field][:,:], threshold=0.7)
        adj[field][:,:] = nx.to_numpy_array(G[field][j])
        edge_index[field] = edge_idx(adj[field][:,:],threshold=0.7)
        bpdata[field] = add_dims(data[field][:,:],10)
        y[field][:] = j
        y[field] = torch.tensor(y[field][:], dtype=torch.long)
        for i in range(bpdata['L'].shape[2]): # band
            bp = [band[i],band[i+1]]
            bpdata[field][:,:,i] = bandpass(bpdata[field][:,:,i],bp,sample_rate=fs)
            for k in range(bpdata[field].shape[0]): #node
                    low = band[i]
                    high = band[i+1]
                    x[field][k,i] = bandpower(bpdata[field][k,:,i],low,high,fs)
                    x[field] = torch.tensor(x[field][:,:],dtype=torch.float32)
    # Put everything into a datalist, first wil be L then R then Re. 3 per loop. 
    # Need X, edge_indices and y.               
        data_list.append(Data(x=x[field][:,:], edge_index=edge_index[field][:,:], y = y[field] ))
    
    return data_list
################################################################
# Setup Variables
idx = ['L', 'R', 'Re']
band = list(range(0, 41, 4))
band[0]=0.0001
fs = 256
data_list = []

for i in tqdm(range(S1.shape[1])):
    data = preprocess(S1[0,i],idx,band,fs)
    for j in range(2):
        data_list.append(data[j])
