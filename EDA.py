# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis 

Author - Rishan Patel, PhD Student in Bioelectronics Group, UCL.

ITs clear that frequency information is key. Perhaps following in the representation of FBCSP will be important, allow adaption and data redundancy to specific freq bands.

"""

import os
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
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
del subject_number, subject_numbers, mat_fname, mat_contents, data_dir

S1 = subject_data['S39'][:, :]


#%% 

def aggregate_eeg_data(S1):
    """
    Aggregate EEG data for each class.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels, 
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).

    Returns:
        l (ndarray): Aggregated EEG data for class 'L'.
        r (ndarray): Aggregated EEG data for class 'R'.
        re (ndarray): Aggregated EEG data for class 'Re'.
    """
    idx = ['L', 'R', 'Re']
    max_sizes = {field: 0 for field in idx}

    # Find the maximum size of EEG data for each class
    for field in idx:
        for i in range(S1[field].shape[1]):
            max_sizes[field] = max(max_sizes[field], S1[field][0, i].shape[0])

    # Initialize arrays to store aggregated EEG data
    l = np.zeros((max_sizes['L'], 22, S1['L'].shape[1]))
    r = np.zeros((max_sizes['R'], 22, S1['R'].shape[1]))
    re = np.zeros((max_sizes['Re'], 22, S1['Re'].shape[1]))

    # Loop through each sample
    for i in range(S1['L'].shape[1]):
        for j, field in enumerate(idx):
            x = S1[field][0, i]  # EEG data for the current sample
            # Resize x to match the maximum size
            resized_x = np.zeros((max_sizes[field], 22))
            resized_x[:x.shape[0], :] = x
            # Add the resized EEG data to the respective array
            if field == 'L':
                l[:, :, i] += resized_x
            elif field == 'R':
                r[:, :, i] += resized_x
            elif field == 'Re':
                re[:, :, i] += resized_x

    return l, r, re

l,r,re = aggregate_eeg_data(S1)

def compute_periodogram(l, r, re, fs=256):
    """
    Compute the periodogram for EEG data from three classes.

    Parameters:
        l (ndarray): EEG data for class 'L' of shape (n_samples, 22, n_instances).
        r (ndarray): EEG data for class 'R' of shape (n_samples, 22, n_instances).
        re (ndarray): EEG data for class 'Re' of shape (n_samples, 22, n_instances).
        fs (int, optional): Sampling frequency. Default is 256 Hz.

    Returns:
        l_f (ndarray): Frequencies for class 'L' of shape (n_freq_bins, n_instances).
        l_Pxx (ndarray): Power spectral density for class 'L' of shape (n_freq_bins, 22, n_instances).
        r_f (ndarray): Frequencies for class 'R' of shape (n_freq_bins, n_instances).
        r_Pxx (ndarray): Power spectral density for class 'R' of shape (n_freq_bins, 22, n_instances).
        re_f (ndarray): Frequencies for class 'Re' of shape (n_freq_bins, n_instances).
        re_Pxx (ndarray): Power spectral density for class 'Re' of shape (n_freq_bins, 22, n_instances).
    """
    pxx = [0, 0, 0]

    # Compute periodogram for each class
    lf, lPxx = sp.signal.periodogram(l[:, 0, 0], fs=fs)
    pxx[0] = lPxx.shape[0]

    rf, rPxx = sp.signal.periodogram(r[:, 0, 0], fs=fs)
    pxx[1] = rPxx.shape[0]

    ref, rePxx = sp.signal.periodogram(re[:, 0, 0], fs=fs)
    pxx[2] = rePxx.shape[0]

    # Preallocate arrays for power spectral density and frequencies
    l_Pxx = np.zeros((pxx[0], 22, l.shape[-1]))
    r_Pxx = np.zeros((pxx[1], 22, r.shape[-1]))
    re_Pxx = np.zeros((pxx[2], 22, re.shape[-1]))

    l_f = np.zeros((pxx[0]))
    r_f = np.zeros((pxx[1]))
    re_f = np.zeros((pxx[2]))

    # Compute periodogram for each instance
    for i in range(l.shape[2]):
        for j in range(22):
            lf, l_Pxx[:, j, i] = sp.signal.periodogram(l[:, j, i], fs=fs)
            l_f = lf

            rf, r_Pxx[:, j, i] = sp.signal.periodogram(r[:, j, i], fs=fs)
            r_f = rf

            ref, re_Pxx[:, j, i] = sp.signal.periodogram(re[:, j, i], fs=fs)
            re_f = ref

    return l_f, l_Pxx, r_f, r_Pxx, re_f, re_Pxx

l_f, l_Pxx, r_f, r_Pxx, re_f, re_Pxx = compute_periodogram(l, r, re,fs=256)

# for i in range(22):
#     plt.figure()
#     plt.semilogy(r_f, np.sqrt(r_Pxx[:,i,0]))
#     plt.ylim([1e-1, 2e3])
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('Linear spectrum [V RMS]')
#     plt.show()

#%%

m_lpxx  = np.mean(l_Pxx, axis=2)
m_rpxx  = np.mean(r_Pxx, axis=2)
m_repxx  = np.mean(re_Pxx, axis=2)

l_f = l_f[:230]
l_Pxx = l_Pxx[:230,:,:]
m_lpxx = m_lpxx[:230,:]

r_f = r_f[:452]
r_Pxx = r_Pxx[:452,:,:]
m_rpxx = m_rpxx[:452,:]


re_f = re_f[:452]
re_Pxx = re_Pxx[:452,:,:]
m_repxx = m_repxx[:452,:]


# Plot for l_Pxx
plt.figure(figsize=(10, 6))
plt.plot(l_f, np.sqrt(m_lpxx))
plt.title('Average Power Spectral Density for Class L')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Linear Spectrum (V RMS)')
plt.grid(True)

# Plot for r_Pxx
plt.figure(figsize=(10, 6))
plt.plot(r_f, np.sqrt(m_rpxx))
plt.title('Average Power Spectral Density for Class R')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Linear Spectrum (V RMS)')
plt.grid(True)

# Plot for re_Pxx
plt.figure(figsize=(10, 6))
plt.plot(re_f, np.sqrt(m_repxx))
plt.title('Average Power Spectral Density for Class Re')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Linear Spectrum (V RMS)')
plt.grid(True)

plt.show()

#%%
m_l  = np.mean(l, axis=2)
m_r  = np.mean(r, axis=2)
m_re  = np.mean(re, axis=2)


# Plot for l_Pxx
plt.figure(figsize=(10, 6))
plt.plot(m_l)
plt.title('Average EEG for Class L')
plt.xlabel('Sample')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# Plot for r_Pxx
plt.figure(figsize=(10, 6))
plt.plot(m_r)
plt.title('Average EEG for Class R')
plt.xlabel('Sample')
plt.ylabel('Amplitude (V)')
plt.grid(True)

# Plot for re_Pxx
plt.figure(figsize=(10, 6))
plt.plot(m_re)
plt.title('Average EEG for Class Re')
plt.xlabel('Sample')
plt.ylabel('Amplitude (V)')
plt.grid(True)

plt.show()

