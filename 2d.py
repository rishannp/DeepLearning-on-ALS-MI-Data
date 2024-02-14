# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
2D CNN using raw EEG data
"""
import stft
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

# %%
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)
 # %% Preparing images
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


def reshape_arrays(S1):
    idx = ['L', 'R', 'Re']
    l = np.transpose(S1['L'][0, 0])
    r = np.transpose(S1['R'][0, 0])
    re = np.transpose(S1['Re'][0, 0])

    for i in range(1, np.size(S1['L']) - 1):
        l = np.concatenate((l, np.transpose(S1['L'][0, i])), axis=1)
        r = np.concatenate((r, np.transpose(S1['R'][0, i])), axis=1)
        re = np.concatenate((re, np.transpose(S1['Re'][0, i])), axis=1)

    ldim = int(np.floor(np.size(l, 1) / 256))
    l = l[:, :ldim * 256]

    rdim = int(np.floor(np.size(r, 1) / 256))
    r = r[:, :rdim * 256]

    redim = int(np.floor(np.size(re, 1) / 256))
    re = re[:, :redim * 256]

    l = l.reshape(22, 256, ldim)
    r = r.reshape(22, 256, rdim)
    re = re.reshape(22, 256, redim)

    return l, r, re


l, r, re = reshape_arrays(subject_data['S1'][:,:])


yl = np.zeros([22,256,815])
yr = np.ones([22,256,779])
yre = np.ones([22,256,258])*2

img = np.concatenate((l,r,re),2)
y = np.concatenate((yl,yr,yre),2)

del yl,yr,yre,l,r,re

# %% Data Augmentation

# %% Data Splitting #2 (CURRENT)

def split(x, y, train_percentage):
    # Shuffle the indices along the 4th dimension
    num_samples = np.size(x,2)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Shuffle both S1 and y using the shuffled indices
    x_shuffled = x[:, :,indices]
    y_shuffled = y[:, :,indices]
    
    # Calculate the number of samples for training
    num_train_samples = int(train_percentage * num_samples)
    
    # Split the shuffled data into training and testing sets
    train_x = x_shuffled[:, :, :num_train_samples]
    train_y = y_shuffled[:, :, :num_train_samples]
    test_x = x_shuffled[:, :, num_train_samples:]
    test_y = y_shuffled[:, :, num_train_samples:]

    # Add an extra dimension of size 1 to each array
    train_x = np.expand_dims(train_x, axis=-1)
    train_x = np.repeat(train_x, 3, axis=-1)  # Repeat along the new dimension to make size 3
    train_y = np.expand_dims(train_y, axis=-1)
    train_y = np.repeat(train_y, 3, axis=-1)  # Repeat along the new dimension to make size 3
    test_x = np.expand_dims(test_x, axis=-1)
    test_x = np.repeat(test_x, 3, axis=-1)  # Repeat along the new dimension to make size 3
    test_y = np.expand_dims(test_y, axis=-1)
    test_y = np.repeat(test_y, 3, axis=-1)  # Repeat along the new dimension to make size 3

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = split(img, y, 0.3)
del img, y

# %% Data Loader 
# Dataloader is jarring asf so you need to have the sample dimension as the first, so i need to reshape it as follows:

# Reshape train_S1 and train_y to have a shape of (n, 22, 64, 156,3)
# Transpose train_S1 and train_y
train_x = np.transpose(train_x, (2, 0, 1, 3))  # Move the sample dimension to the first dimension
train_y = np.transpose(train_y, (2, 0, 1, 3))    # Adjust the dimensions as needed for labels
# Transpose test_S1 and test_y (assuming you intended to transpose test_S1)
test_x = np.transpose(test_x, (2, 0, 1, 3))  # Move the sample dimension to the first dimension
test_y = np.transpose(test_y, (2, 0, 1, 3))    # Adjust the dimensions as needed for labels


train_loader = tf.data.Dataset.from_tensor_slices((train_x, train_y))
validation_loader = tf.data.Dataset.from_tensor_slices((test_x, test_y))
batch_size = 2


train_dataset = (
    train_loader.prefetch(2)
)

validation_dataset = (
    validation_loader.prefetch(2)
)

# %% EfficientNetv2

class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [22, 256]  # Input image size
    epochs = 100 # Training epochs
    batch_size = 3  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 3 # Number of classes in the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['L', 'R', 'Re']
    label2name = dict(enumerate(class_names))
    name2label = {v:k for k, v in label2name.items()}

#new_input = layers.Input(shape=(22, 256, 1),name='image_input')
LOSS = keras.losses.KLDivergence()
    
model = keras_cv.models.ImageClassifier.from_preset(
    CFG.preset, num_classes=CFG.num_classes,input_shape=(22,256,3)
)

# Compile the model  
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=LOSS)

# Model Sumamry
model.summary()

# LR Schedule
def get_lr_callback(batch_size=3, mode='cos', epochs=50, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

lr_cb = get_lr_callback(CFG.batch_size, mode=CFG.lr_mode, plot=True)

# Checkpoint to find best models
ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras",
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='min')
# %%

history = model.fit(
    train_dataset, 
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb], 
    steps_per_epoch=len(train_dataset)//CFG.batch_size,
    validation_data=validation_dataset, 
    verbose=CFG.verbose
)
