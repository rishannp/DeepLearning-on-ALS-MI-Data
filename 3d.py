# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
3D CNN using spectrogram images
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
from sklearn.preprocessing import OneHotEncoder
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

    yl = np.zeros([1,160])
    yr = np.ones([1,160])
    yre = np.ones([1,160])*2

    img = np.concatenate((l,r,re),3)
    y = np.concatenate((yl,yr,yre),1)
    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)
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


# %% Data Splitting #2 (CURRENT)

def split(S1, y, train_percentage):
    # Shuffle the indices along the 4th dimension
    num_samples = np.size(S1, 3)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Shuffle both S1 and y using the shuffled indices
    S1_shuffled = S1[:, :, :, indices]
    y_shuffled = y[indices, :]
    
    # Calculate the number of samples for training
    num_train_samples = int(train_percentage * num_samples)
    
    # Split the shuffled data into training and testing sets
    train_S1 = S1_shuffled[:, :, :, :num_train_samples]
    train_y = y_shuffled[:num_train_samples, :]
    test_S1 = S1_shuffled[:, :, :, num_train_samples:]
    test_y = y_shuffled[num_train_samples:, :]

    train_S1 = np.expand_dims(train_S1, axis=-1)
    train_S1 = np.repeat(train_S1, 3, axis=-1)  # Repeat along the new dimension to make size 3

    test_S1 = np.expand_dims(test_S1, axis=-1)
    test_S1 = np.repeat(test_S1, 3, axis=-1)  # Repeat along the new dimension to make size 3


    return train_S1, train_y, test_S1, test_y

train_S1, train_y, test_S1, test_y = split(S1, y, 0.3)
del S1, y

# %% Data Loader 
# Dataloader is jarring asf so you need to have the sample dimension as the first, so i need to reshape it as follows:

# Reshape train_S1 and train_y to have a shape of (n, 22, 64, 156,1)
# Transpose train_S1 and train_y
train_S1 = np.transpose(train_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension
#train_y = np.transpose(train_y, (3, 0, 1, 2,4))    # Adjust the dimensions as needed for labels
# Transpose test_S1 and test_y (assuming you intended to transpose test_S1)
test_S1 = np.transpose(test_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension
#test_y = np.transpose(test_y, (3, 0, 1, 2,4))    # Adjust the dimensions as needed for labels


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

# %% EfficientNetv2

class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [22, 64, 156]  # Input image size
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
    CFG.preset, num_classes=CFG.num_classes,input_shape=(22,64,156,3)
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
# %% Training (takes a while with 100 epochs)

history = model.fit(
    train_dataset, 
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb], 
    steps_per_epoch=len(train_dataset)//CFG.batch_size,
    validation_data=validation_dataset, 
    verbose=CFG.verbose
)

# %% 

model.load_weights("best_model.keras") # Finds optimal model from training
preds = model.predict(validation_dataset)