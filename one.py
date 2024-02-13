# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
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


def makespect(S1):
    # Find max size of all classes to allow proper preallocation
    idx = ['L', 'R', 'Re']
     
    dtype = [('L', np.float32), ('R', np.float32), ('Re', np.float32)]
    
    # Create an empty structured array with the defined data type
    img = np.zeros((22, 64, 156, np.size(S1, 1)), dtype=dtype)
    
    # S1 now has a full band of X examples for each class (L, R, Re)
    # Create spectrogram for each channel
    for i in range(np.size(S1, 1)-1):
        for j, field in enumerate(idx):
            x = np.transpose(S1[field][0, i])
            
            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=256, hop_length=np.size(x, 1) // 64, 
                                                      n_fft=256, n_mels=64, fmin=0, fmax=100, win_length=128)
            
            # LOG TRANSFORM
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :, :width]
            
            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db + 40) / 40 
            
            size_of_mel_spec_db = np.size(mel_spec_db, 2)
            #img[idx[j]][:,:,:,i] += mel_spec_db
            img[field][:, :, :size_of_mel_spec_db,i] += mel_spec_db
            
            if i and j != 0:
                print( (i+j)/(i*j) * 100)
    
    l = img['L'][:,:,:,:]
    r = img['R'][:,:,:,:]
    re = img['Re'][:,:,:,:]

    yl = np.zeros([22,64,156,160])
    yr = np.ones([22,64,156,160])
    yre = np.ones([22,64,156,160])*3

    img = np.concatenate((l,r,re),3)
    y = np.concatenate((yl,yr,yre),3)
    return img,y

# I have had to prepad the third dimension with zeros as not all classes or instances are the same time length. Therefore many images will show some no activity areas. 
S1,y = makespect(subject_data['S1'])
# S2 = makespect(subject_data['S2'])
# S5 = makespect(subject_data['S5'])
# S9 = makespect(subject_data['S9'])
# S21 = makespect(subject_data['S21'])
# S31 = makespect(subject_data['S31'])
# S34 = makespect(subject_data['S34'])
# S39 = makespect(subject_data['S39'])


#%%
x = S1[:,:,:,1]
# RAW SPECTROGRAM
mel_spec = librosa.feature.melspectrogram(y=x, sr=256, hop_length=np.size(x,1)//64, 
                                          n_fft=256, n_mels=64, fmin=0, fmax=100, win_length=16)

# The output of mel_spec is 22x64x68. That is Channels X Mel Bin Height X Number of Frames
# Number of frames is calculated by (size(x,1)-n_fft) // hop_length + 1. so 1080-256 // 16+1 = 68
 
# LOG TRANSFORM
width = (mel_spec.shape[1]//32)*32
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

# STANDARDIZE TO -1 TO 1
mel_spec_db = (mel_spec_db+40)/40 
   

plt.figure()
plt.imshow(mel_spec_db[:,:,0,0])
plt.colorbar()
# # np.savez("spectrogram_data.npz", S1=S1, S2=S2, S5=S5, S9=S9, S21=S21, S31=S31, S34=S34, S39=S39)
# %% Data Augmentation


 
# %% Data splitting

# def split1(S1, y, train_percentage):
#     # Shuffle the indices along the 4th dimension
#     num_samples = np.size(S1, 3)
#     indices = np.arange(num_samples)
#     np.random.shuffle(indices)

#     # Shuffle both S1 and y using the shuffled indices
#     S1_shuffled = S1[:, :, :, indices]
#     y_shuffled = y[:, :, :, indices]
    
#     # Calculate the number of samples for training
#     num_train_samples = int(train_percentage * num_samples)
    
#     # Split the shuffled data into training and testing sets
#     train_S1 = S1_shuffled[:, :, :, :num_train_samples]
#     train_y = y_shuffled[:, :, :, :num_train_samples]
#     test_S1 = S1_shuffled[:, :, :, num_train_samples:]
#     test_y = y_shuffled[:, :, :, num_train_samples:]

#     return train_S1, train_y, test_S1, test_y


# train_S1, train_y, test_S1, test_y = split1(S1, y, 0.3)
# del S1,y

# %% 

def split(S1, y, train_percentage):
    # Shuffle the indices along the 4th dimension
    num_samples = np.size(S1, 3)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Shuffle both S1 and y using the shuffled indices
    S1_shuffled = S1[:, :, :, indices]
    y_shuffled = y[:, :, :, indices]
    
    # Calculate the number of samples for training
    num_train_samples = int(train_percentage * num_samples)
    
    # Split the shuffled data into training and testing sets
    train_S1 = S1_shuffled[:, :, :, :num_train_samples]
    train_y = y_shuffled[:, :, :, :num_train_samples]
    test_S1 = S1_shuffled[:, :, :, num_train_samples:]
    test_y = y_shuffled[:, :, :, num_train_samples:]

    # Add an extra dimension of size 1 to each array
    train_S1 = np.expand_dims(train_S1, axis=-1)
    train_y = np.expand_dims(train_y, axis=-1)
    test_S1 = np.expand_dims(test_S1, axis=-1)
    test_y = np.expand_dims(test_y, axis=-1)

    return train_S1, train_y, test_S1, test_y

train_S1, train_y, test_S1, test_y = split(S1, y, 0.3)
del S1, y

# %%

# %% Data Loader 
# Dataloader is jarring asf so you need to have the sample dimension as the first, so i need to reshape it as follows:

# Reshape train_S1 and train_y to have a shape of (n, 22, 64, 156,1)
# Transpose train_S1 and train_y
train_S1 = np.transpose(train_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension
train_y = np.transpose(train_y, (3, 0, 1, 2,4))    # Adjust the dimensions as needed for labels
# Transpose test_S1 and test_y (assuming you intended to transpose test_S1)
test_S1 = np.transpose(test_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension
test_y = np.transpose(test_y, (3, 0, 1, 2,4))    # Adjust the dimensions as needed for labels


train_loader = tf.data.Dataset.from_tensor_slices((train_S1, train_y))
validation_loader = tf.data.Dataset.from_tensor_slices((test_S1, test_y))
batch_size = 2

# Augment the on the fly during training.
train_dataset = (
    train_loader.prefetch(2)
)

validation_dataset = (
    validation_loader.prefetch(2)
)

# %% Test take batch 

# for batch in train_dataset.take(1):
#     # 'batch' will be a tuple containing input data and labels
#     input_data, labels = batch

#     # Print the shapes of input data and labels
#     print("Input data shape:", input_data.shape)
#     print("Labels shape:", labels.shape)
    

# %% Model loading

def get_model(batch_size, width, height, depth):
    """Build a 3D convolutional neural network model."""


    # Define the input shape
    #input_shape = (None, width, height, depth, 1)]
    #inputs = layers.Input(input_shape)
    #inputs = keras.Input((width, height, depth, 1))
    
    input_shape = (batch_size,width,height,depth,1)
    input_shape = tf.random.normal(input_shape)
    
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu",input_shape=input_shape[1:])(input_shape)
    print(x.shape) # Input shape: (22, 64, 156, 1), Output shape: (20, 62, 154, 32)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    print(x.shape) # Input shape: (20, 62, 154, 32), Output shape: (10, 31, 77, 32)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu")(x)
    print(x.shape) # Input shape: (10, 31, 77, 32), Output shape: (8, 29, 75, 64)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    print(x.shape) # Input shape: (8, 29, 75, 64), Output shape: (4, 14, 37, 64)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(x)
    print(x.shape) # Input shape: (4, 14, 37, 64), Output shape: (2, 12, 35, 128)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=256, kernel_size=(1, 1, 1), activation="relu")(x)
    print(x.shape) # Input shape: (2, 12, 35, 128), Output shape: (2, 12, 35, 256)
    x = layers.BatchNormalization()(x)
    # Global average pooling layer
    x = layers.GlobalAveragePooling3D()(x)
    print(x.shape) # Input shape: (2, 12, 35, 256), Output shape: (256,)
    # Fully connected layers
    x = layers.Dense(units=512, activation="relu")(x)
    print(x.shape) # Input shape: (256,), Output shape: (512,)
    # Dropout layer
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(units=3, activation="softmax")(x)
    # Input shape: (512,), Output shape: (3,)

    # Define the model.
    model = keras.Model(input_shape, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(batch_size = 1, width=22, height=64, depth=156)
model.summary()


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
    run_eagerly=True,
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.keras", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
