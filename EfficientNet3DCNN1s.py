# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
3D CNN based on EfficientNetV2 using spectrogram images. : UNFINISHED _ 
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


S1 = subject_data['S1']
batch_size = 1
epoch = 10
train_p = 0.3
val_p = 0.2
test_p = 0.5

# %% 

def reshape_arrays(S1):
    idx = ['L', 'R', 'Re']
    l_list = [np.transpose(S1[key][0, 0]) for key in idx]
    for i in range(1, np.size(S1[idx[0]]) - 1):
        l_list = [np.concatenate((l, np.transpose(S1[key][0, i])), axis=1) for l, key in zip(l_list, idx)]

    l_list = [l[:, :(int(np.size(l, 1) / 256)) * 256] for l in l_list]
    l_list = [l.reshape(22, 256, int(np.size(l, 1) / 256)) for l in l_list]

    y_list = [np.zeros([np.size(l, 2), 1]) for l in l_list]
    y_list[1] = np.ones([np.size(l_list[1], 2), 1])
    y_list[2] = np.ones([np.size(l_list[2], 2), 1]) * 2

    img = np.concatenate(l_list, axis=2)
    y = np.concatenate(y_list)
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)

    return img, y

img, y = reshape_arrays(subject_data['S1'][:,:])


# % Data Augmentation (To be added)

# Converting to spect

def makespect(img):
    img_shape = img.shape
    img1 = np.zeros((22, 64, 64, img_shape[2]), dtype=np.float32)

    for i in range(img_shape[2]):
        x = img[:, :, i]

        mel_spec = librosa.feature.melspectrogram(y=x, sr=256, hop_length=img_shape[1] // 64,
                                                  n_fft=256, n_mels=64, fmin=0, fmax=100, win_length=128)

        width = (mel_spec.shape[1] // 32) * 32
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :, :width]
        mel_spec_db = (mel_spec_db + 40) / 40

        img1[:, :, :, i] = mel_spec_db

    return img1


img = makespect(img)

# Make train val test 
def split(S1, y, train_p, val_p, test_p):
    idx = np.zeros([2, 3])
    idx[0, 0] = np.argmax(y[:, 0])
    idx[1, 0] = np.argmin(y[:, 0])

    idx[0, 1] = np.argmin(y[:, 0])
    idx[1, 1] = np.argmax(y[:, 2])

    idx[0, 2] = np.argmax(y[:, 2])
    idx[1, 2] = np.size(y, 0) - 1

    C1 = np.arange(int(idx[0, 0]), int(idx[1, 0]))
    C2 = np.arange(int(idx[0, 1]), int(idx[1, 1]))
    C3 = np.arange(int(idx[0, 2]), int(idx[1, 2]))

    np.random.shuffle(C1)
    np.random.shuffle(C2)
    np.random.shuffle(C3)

    train_size_C1 = int(train_p * len(C1))
    val_size_C1 = int(val_p * len(C1))

    train_size_C2 = int(train_p * len(C2))
    val_size_C2 = int(val_p * len(C2))

    train_size_C3 = int(train_p * len(C3))
    val_size_C3 = int(val_p * len(C3))

    train_C1, val_C1, test_C1 = C1[:train_size_C1], C1[train_size_C1:train_size_C1 + val_size_C1], C1[train_size_C1 + val_size_C1:]
    train_C2, val_C2, test_C2 = C2[:train_size_C2], C2[train_size_C2:train_size_C2 + val_size_C2], C2[train_size_C2 + val_size_C2:]
    train_C3, val_C3, test_C3 = C3[:train_size_C3], C3[train_size_C3:train_size_C3 + val_size_C3], C3[train_size_C3 + val_size_C3:]

    train_indices = np.concatenate([train_C1, train_C2, train_C3])
    val_indices = np.concatenate([val_C1, val_C2, val_C3])
    test_indices = np.concatenate([test_C1, test_C2, test_C3])

    train_S1 = S1[:, :, :, train_indices]
    train_y = y[train_indices, :]

    val_S1 = S1[:, :, :, val_indices]
    val_y = y[val_indices, :]

    test_S1 = S1[:, :, :, test_indices]
    test_y = y[test_indices, :]

    # Repeat and expand dimensions for S1 to make size 3
    train_S1 = np.expand_dims(train_S1, axis=-1)
    train_S1 = np.repeat(train_S1, 3, axis=-1)
    val_S1 = np.expand_dims(val_S1, axis=-1)
    val_S1 = np.repeat(val_S1, 3, axis=-1)
    test_S1 = np.expand_dims(test_S1, axis=-1)
    test_S1 = np.repeat(test_S1, 3, axis=-1)

    return train_S1, train_y, val_S1, val_y, test_S1, test_y

train_S1, train_y, val_S1, val_y, test_S1, test_y = split(img, y, 0.3, 0.2, 0.5)

# %% Model

# Edit data size
# Inefficient code

# def reshape_arrays(S1):
#     idx = ['L', 'R', 'Re']
#     l = np.transpose(S1['L'][0, 0])
#     r = np.transpose(S1['R'][0, 0])
#     re = np.transpose(S1['Re'][0, 0])

#     for i in range(1, np.size(S1['L']) - 1):
#         l = np.concatenate((l, np.transpose(S1['L'][0, i])), axis=1)
#         r = np.concatenate((r, np.transpose(S1['R'][0, i])), axis=1)
#         re = np.concatenate((re, np.transpose(S1['Re'][0, i])), axis=1)

#     ldim = int(np.floor(np.size(l, 1) / 256))
#     l = l[:, :ldim * 256]

#     rdim = int(np.floor(np.size(r, 1) / 256))
#     r = r[:, :rdim * 256]

#     redim = int(np.floor(np.size(re, 1) / 256))
#     re = re[:, :redim * 256]

#     l = l.reshape(22, 256, ldim)
#     r = r.reshape(22, 256, rdim)
#     re = re.reshape(22, 256, redim)
    
#     yl = np.zeros([np.size(l,2),1])
#     yr = np.ones([np.size(r,2),1])
#     yre = np.ones([np.size(re,2),1])*2

#     img = np.concatenate((l,r,re),2)
#     y = np.concatenate((yl,yr,yre))
#     encoder = OneHotEncoder(sparse=False)
#     y = encoder.fit_transform(y)
    
#     return img, y




# img,y = reshape_arrays(subject_data['S1'][:,:])

# % Data Augmentation (To be added)

# Converting to spect

# def makespect(img):

#     # Create an empty structured array with the defined data type
#     img1 = np.zeros((22, 64, 64, np.size(img, 2)))
    
#     # S1 now has a full band of X examples for each class (L, R, Re)
#     # Create spectrogram for each channel
#     for i in range(np.size(img, 2)-1):
#             x = img[:,:,i]
            
#             # RAW SPECTROGRAM
#             mel_spec = librosa.feature.melspectrogram(y=x, sr=256, hop_length=np.size(x, 1) // 64, 
#                                                       n_fft=256, n_mels=64, fmin=0, fmax=100, win_length=128)
            
#             # LOG TRANSFORM
#             width = (mel_spec.shape[1] // 32) * 32
#             mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :, :width]
            
#             # STANDARDIZE TO -1 TO 1
#             mel_spec_db = (mel_spec_db + 40) / 40 
            
#             #img[idx[j]][:,:,:,i] += mel_spec_db
#             img1[:,:,:,i] += mel_spec_db
            
    
#     return img1

 
# img = makespect(img)

# % Split data

# def split(S1, y, train_p, val_p, test_p):
#     idx = np.zeros([2,3])
#     idx[0,0] = np.argmax(y[:,0])
#     idx[1,0] = np.argmin(y[:,0])     
    
#     idx[0,1] = np.argmin(y[:,0])
#     idx[1,1] = np.argmax(y[:,2])
    
#     idx[0,2] = np.argmax(y[:,2])
#     idx[1,2] = np.size(y,0)-1

#     C1 = np.array([i for i in range(int(idx[0,0]), int(idx[1,0]))])
#     C2 = np.array([i for i in range(int(idx[0,1]), int(idx[1,1]))])
#     C3 = np.array([i for i in range(int(idx[0,2]), int(idx[1,2]))])
    
#     nc1 = np.size(C1, -1)
#     ic1 = np.arange(nc1)
#     np.random.shuffle(ic1)

#     nc2 = np.size(C2, -1)
#     ic2 = np.arange(nc2)
#     np.random.shuffle(ic2)

#     nc3 = np.size(C3, -1)
#     ic3 = np.arange(nc3)
#     np.random.shuffle(ic3)
    
#     C1 = C1[ic1,]
#     C2 = C1[ic2,]
#     C3 = C1[ic3,]
    
#     C1train = C1[:int(train_p * np.size(C1,0)),]
#     C1val = C1[int(train_p * np.size(C1,0)):int(train_p * np.size(C1,0))+int(val_p * np.size(C1,0)),]
#     C1test = C1[int(train_p * np.size(C1,0))+int(val_p * np.size(C1,0)):int(train_p * np.size(C1,0))+int(val_p * np.size(C1,0))+int(test_p * np.size(C1,0)),]
    
#     C2train = C2[:int(train_p * np.size(C2,0)),]
#     C2val = C2[int(train_p * np.size(C2,0)):int(train_p * np.size(C2,0))+int(val_p * np.size(C2,0)),]
#     C2test = C2[int(train_p * np.size(C2,0))+int(val_p * np.size(C2,0)):int(train_p * np.size(C2,0))+int(val_p * np.size(C2,0))+int(test_p * np.size(C2,0)),]
    
#     C3train = C3[:int(train_p * np.size(C3,0)),]
#     C3val = C3[int(train_p * np.size(C3,0)):int(train_p * np.size(C3,0))+int(val_p * np.size(C3,0)),]
#     C3test = C3[int(train_p * np.size(C3,0))+int(val_p * np.size(C3,0)):int(train_p * np.size(C3,0))+int(val_p * np.size(C3,0))+int(test_p * np.size(C3,0)),]


#     # Need to now just make all the train test val and then concat them on the column dim
#     train = np.concatenate([C1train,C2train,C3train])
#     val = np.concatenate([C1val,C2val,C3val])
#     test = np.concatenate([C1test,C2test,C3test])
    
#     train_S1 = img[:,:,:,train]
#     train_y = y[train,:]
    
#     val_S1 = img[:,:,:,val]
#     val_y = y[val,:]
    
#     test_S1 = img[:,:,:,test]
#     test_y = y[test,:]
    
#     # Repeat and expand dimensions for S1 to make size 3
#     train_S1 = np.expand_dims(train_S1, axis=-1)
#     train_S1 = np.repeat(train_S1, 3, axis=-1)
#     val_S1 = np.expand_dims(val_S1, axis=-1)
#     val_S1 = np.repeat(val_S1, 3, axis=-1)
#     test_S1 = np.expand_dims(test_S1, axis=-1)
#     test_S1 = np.repeat(test_S1, 3, axis=-1)

#     return train_S1, train_y, val_S1, val_y, test_S1, test_y

# train_S1, train_y, val_S1, val_y, test_S1, test_y = split(S1, y, 0.3, 0.2, 0.5)

# %% Data Loader 
# Dataloader is jarring asf so you need to have the sample dimension as the first, so i need to reshape it as follows:

# Reshape train_S1 and train_y to have a shape of (n, 22, 64, 156,3)
train_S1 = np.transpose(train_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension
val_S1 = np.transpose(val_S1, (3, 0, 1, 2,4))
test_S1 = np.transpose(test_S1, (3, 0, 1, 2,4))  # Move the sample dimension to the first dimension

train_loader = tf.data.Dataset.from_tensor_slices((train_S1, train_y)) # could add repeat
validation_loader = tf.data.Dataset.from_tensor_slices((val_S1, val_y))
test_loader = tf.data.Dataset.from_tensor_slices((test_S1, test_y))

train_dataset = (
    train_loader.prefetch(batch_size)
)

validation_dataset = (
    validation_loader.prefetch(batch_size)
)

test_dataset = (
    test_loader.prefetch(batch_size)
)

# EfficientNetv2

class CFG: #Configure
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [22, 64, 156]  # Input image size
    epochs = 10 # Training epochs
    batch_size = 1  # Batch size
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
def get_lr_callback(batch_size=batch_size, mode='cos', epochs=epoch, plot=False):
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
# Training (takes a while with 100 epochs)

history = model.fit(
    train_dataset, 
    epochs=CFG.epochs,
    callbacks=[lr_cb, ckpt_cb], 
    steps_per_epoch=len(train_dataset)//CFG.batch_size,
    validation_data=validation_dataset, 
    verbose=CFG.verbose
)

# Prediction

model.load_weights("best_model.keras") # Finds optimal model from training
preds = model.predict(test_dataset)

#pred = np.zeros([1,3,336])
labelout = np.zeros([1,np.size(preds,0) // 22])

for i in range(np.size(labelout,1)-1):
    x = preds[i*22:(i+1)*22, :]
    x = np.mean(x,axis=0)
    #pred[:,:,i] += x
    labelout[:,i] = np.argmax(x)
    #max_value = np.amax(column_means)
labelout = np.transpose(labelout)

encoder = OneHotEncoder(sparse=False)
labelout = encoder.fit_transform(labelout)

# Check if dimension 3 exists
if len(labelout.shape) > 3:
    print("Dimension 3 exists.")
else:
    print("Dimension 3 does not exist.")

# Check the size of the second dimension
if labelout.shape[1] != 3:
    # Calculate the number of rows in the array
    rows = labelout.shape[0]

    # Create a new array with the desired shape (rows, 3)
    padded_labelout = np.ones((rows, 3))*3

    # Copy the existing values into the new array
    padded_labelout[:, :labelout.shape[1]] = labelout

    # Update the original array with the padded array
    labelout = padded_labelout

    print("Array padded with zeros to make it (rows, 3).")
else:
    print("Second dimension is already size 3.")

accuracies = np.zeros((3,))

# Iterate over each class
for i in range(3):
    # Extract the one-hot encoded labels for the current class
    true_labels = test_y[:, i]
    predicted_labels = labelout[:, i]

    # Calculate the number of correct predictions for the current class
    correct_predictions = np.sum(true_labels == predicted_labels)

    # Calculate the total number of samples for the current class
    total_samples = len(true_labels)

    # Calculate the accuracy for the current class
    accuracy = correct_predictions / total_samples

    # Store the accuracy in the array
    accuracies[i] = accuracy
    
    
# Output the accuracies for each class
print("Accuracies for each class:")
print(accuracies)



# accuracies, model = makemodel(subject_data['S2'][:,:],1,20,0.3,0.2,0.5)

