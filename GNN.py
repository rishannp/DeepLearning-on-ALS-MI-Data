# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.

Building a dataset and testing a GNN model. 
Poor accuracies sub 0.4.
Using PLV from EEG signals, create graphs. the feature vector has Spectral Entropy, PSD, and Relative Power.
Need to improve my feature engineering. 

https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
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
import torch as torch
from scipy.signal import welch
from scipy.stats import entropy
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

S1 = subject_data['S39'][:,:]

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
    plv = {field: np.zeros((22, 22, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r, re = plv['L'], plv['R'], plv['Re']
    yl, yr, yre = np.zeros((subject_data.shape[1], 1)), np.ones((subject_data.shape[1], 1)), np.full((subject_data.shape[1], 1), 2)
    img = np.concatenate((l, r, re), axis=2)
    y = np.concatenate((yl, yr, yre), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y

plv, y = compute_plv(S1)

def create_graphs(plv, threshold=0.5):
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
for i, G in enumerate(graphs):
    adj[:, :, i] = nx.to_numpy_array(G)

# Initialize empty lists to store node attributes
# node_degrees_list = []
# clustering_coefficient_list = []
# degree_centrality_list = []

# # Extract node attributes for each graph
# for graph in graphs:
#     # Node degrees
#     degrees = np.array(list(dict(graph.degree()).values()))  # Extract node degrees
#     node_degrees_list.append(degrees)

#     # Clustering coefficient
#     clustering_coefficients = np.array(list(nx.clustering(graph).values()))  # Extract clustering coefficients
#     clustering_coefficient_list.append(clustering_coefficients)

#     # Degree centrality
#     centrality = np.array(list(nx.degree_centrality(graph).values()))  # Extract degree centrality
#     degree_centrality_list.append(centrality)

# # Stack node attributes into a tensor
# node_degrees = torch.tensor(node_degrees_list, dtype=torch.float)
# clustering_coefficients = torch.tensor(clustering_coefficient_list, dtype=torch.float)
# degree_centrality = torch.tensor(degree_centrality_list, dtype=torch.float)

def calculate_features(eeg_signal, fs):
    """
    Calculate PSD, Spectral Entropy, and Relative Power for EEG signal.

    Args:
    - eeg_signal (np.ndarray): EEG signal with shape (time_samples, 22).
    - fs (float): Sampling frequency of the EEG signal.

    Returns:
    - concatenated_features (np.ndarray): Concatenated features with shape (3, 22).
    """
    psd = np.zeros((5, eeg_signal.shape[1]))  # Initialize PSD matrix
    spectral_entropy = np.zeros((1, eeg_signal.shape[1]))  # Initialize spectral entropy array
    relative_power = np.zeros((5, eeg_signal.shape[1]))  # Initialize relative power matrix
    
    # Define frequency bands (example)
    freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    
    for t in range(eeg_signal.shape[1]):
        for i, (f_low, f_high) in enumerate(freq_bands):
            f, Pxx = welch(eeg_signal[:, t].reshape(1, -1), fs=fs)  # Welch's method
            idx_band = np.where((f >= f_low) & (f < f_high))[0]  # Indices within frequency band
            avg_power = np.mean(Pxx[:, idx_band], axis=1)  # Average power within frequency band
            psd[i, t] += avg_power.flatten()
            total_power = np.sum(Pxx)  # Total power across all bands
            spectral_entropy[0, t] += entropy(Pxx.T)  # Calculate spectral entropy
            relative_power[i, t] += np.sum(Pxx[0,idx_band]) / total_power  # Relative power
    
    # Average over the time samples
    psd /= eeg_signal.shape[0]
    spectral_entropy /= eeg_signal.shape[0]
    relative_power /= eeg_signal.shape[0]
    
    # Concatenate the features
    concatenated_features = np.concatenate((psd, spectral_entropy, relative_power), axis=0)
    return concatenated_features

    
idx = ['L', 'R', 'Re']
x = {field: np.zeros((11, 22, S1.shape[1])) for field in idx}
for i, field in enumerate(idx):
    for j in range(S1.shape[1]):
        x[field][:,:,j] = calculate_features(S1[field][0,j], 256)

l, r, re = x['L'], x['R'], x['Re']
x = np.concatenate((l, r, re), axis=2)
x = x.transpose([1,0,2])
x = torch.tensor(x,dtype=torch.float32)
# Concatenate node attributes along the feature dimension
#x = torch.cat([node_degrees.unsqueeze(2), clustering_coefficients.unsqueeze(2), degree_centrality.unsqueeze(2)], dim=2)

#del centrality,clustering_coefficient_list,clustering_coefficients,degree_centrality,degree_centrality_list, node_degrees,node_degrees_list,G,graph,S1,plv,subject_data,degrees

#% Initialize an empty list to store edge indices
edge_indices = []

# Iterate over the adjacency matrices
for i in range(adj.shape[2]):
    # Initialize lists to store source and target nodes
    source_nodes = []
    target_nodes = []
    
    # Iterate through each element of the adjacency matrix
    for row in range(adj.shape[0]):
        for col in range(adj.shape[1]):
            # Check if there's an edge
            if adj[row, col, i] == 1:
                # Add source and target nodes to the lists
                source_nodes.append(row)
                target_nodes.append(col)
            else:
                # If no edge exists, add placeholder zeros to maintain size
                source_nodes.append(0)
                target_nodes.append(0)
    
    # Create edge index as a LongTensor
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # Append edge index to the list
    edge_indices.append(edge_index)

# Stack all edge indices along a new axis to create a 2D tensor
edge_indices = torch.stack(edge_indices, dim=-1)

del col,edge_index,i,row,source_nodes,target_nodes

from torch_geometric.data import Data

data_list = []
for i in range(np.size(adj,2)):
    data_list.append(Data(x=x[:, :, i], edge_index=edge_indices[:,:,i], y=y[i, 0]))

def split(data_list, train_p, val_p, test_p):
    num_samples = len(data_list)
    a = num_samples // 3
    class_splits = [
        int(train_p * a), int(train_p * a), int(train_p * a),
        int(val_p * a), int(val_p * a), int(val_p * a),
        int(test_p * a), int(test_p * a), int(test_p * a)
    ]
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    train_indices = indices[:sum(class_splits[:3])]
    val_indices = indices[sum(class_splits[:3]):sum(class_splits[:6])]
    test_indices = indices[sum(class_splits[:6]):]

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]

    return train_data, val_data, test_data


train,val,test = split(data_list,0.5,0,0.5)

del subject_data,S1,re,r,l,plv,field
#%%

# Data loader takes and shows adjacency matrixes in the form of sparse matrices. Saves time and computational resources. Quite cool actually! 
# Figure 5 from - https://blog.dataiku.com/graph-neural-networks-part-three
torch.manual_seed(12345)

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train, batch_size=16, shuffle=True)
val_loader = DataLoader(val, batch_size=16, shuffle=False)
test_loader = DataLoader(test, batch_size=16, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
    
#% GCN

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(11, hidden_channels) #num node features
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 3) # num of classes

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels=16)
print(model)

#%


model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')