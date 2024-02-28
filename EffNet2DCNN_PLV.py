# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.
2D CNN based on EfficientNetV2 using PLV images.
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

# S1 = subject_data['S1'][:,:]

batch_size = 1
epoch = 10

# %% 




# def hilphase(y1,y2): # well behaved sig
#     sig1_hill=sig.hilbert(y1)
#     sig2_hill=sig.hilbert(y2)
#     phase_y1=np.unwrap(np.angle(sig1_hill))
#     phase_y2=np.unwrap(np.angle(sig2_hill))
#     Inst_phase_diff=phase_y1-phase_y2
#     avg_phase=np.average(Inst_phase_diff)
    
#     return Inst_phase_diff

# def hilphase(y1,y2): # Non-behaving sig
#     sig1_hill=sig.hilbert(y1)
#     sig2_hill=sig.hilbert(y2)
#     pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
#                np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
#     phase = np.angle(pdt)

#     return phase


# %% Model

def makemodel(S1,batch_size,epoch,train_p, val_p, test_p): 
    def plvfcn(eegData):
        numElectrodes = eegData.shape[1]
        numTimeSteps = eegData.shape[0]

        # Initialize the PLV matrix
        plvMatrix = np.zeros((numElectrodes, numElectrodes))

        # Calculate PLV matrix for all electrode pairs
        for electrode1 in range(numElectrodes):
            for electrode2 in range(electrode1 + 1, numElectrodes):
                # Calculate the instantaneous phase of the signals
                phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
                phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))

                # Calculate the phase difference between the two signals
                phase_difference = phase2 - phase1

                # Calculate the Phase-Locking Value (PLV)
                plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)

                # Store the PLV value in the matrix (both upper and lower triangular)
                plvMatrix[electrode1, electrode2] = plv
                plvMatrix[electrode2, electrode1] = plv

        return plvMatrix

    def compute_plv(subject_data):
        S1 = subject_data
        idx = ['L', 'R', 'Re']
        plv = {field: np.zeros((22, 22, 160)) for field in idx}

        for i in range(np.size(S1, 1) - 1):
            for j, field in enumerate(idx):
                x = S1[field][0, i]
                plv[field][:, :, i] = plvfcn(x)
                
        l = plv['L'][:,:,:]
        r = plv['R'][:,:,:]
        re = plv['Re'][:,:,:]

        yl = np.zeros([1,np.size(S1,1)])
        yr = np.ones([1,np.size(S1,1)])
        yre = np.ones([1,np.size(S1,1)])*2

        img = np.concatenate((l,r,re),2)
        y = np.concatenate((yl,yr,yre),1)
        y = y.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y)
        
        return img,y

    plv,y = compute_plv(subject_data['S1'][:,:])

    # Data Augmentation (To be added)
    
    # Data Splitting
    
    def split(S1, y, train_p, val_p, test_p):
        # Shuffle the indices along the last dimension
        num_samples = np.size(S1, -1)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
    
        # Shuffle both S1 and y using the shuffled indices
        S1_shuffled = S1[..., indices]
        y_shuffled = y[indices, :]
        
        # Calculate the number of samples for each split
        num_train_samples = int(train_p * num_samples)
        num_val_samples = int(val_p * num_samples)
        num_test_samples = int(test_p * num_samples)
        
        # Split the shuffled data into training, validation, and testing sets
        train_S1 = S1_shuffled[..., :num_train_samples]
        train_y = y_shuffled[:num_train_samples, :]
        val_S1 = S1_shuffled[..., num_train_samples:num_train_samples+num_val_samples]
        val_y = y_shuffled[num_train_samples:num_train_samples+num_val_samples, :]
        test_S1 = S1_shuffled[..., num_train_samples+num_val_samples:num_train_samples+num_val_samples+num_test_samples]
        test_y = y_shuffled[num_train_samples+num_val_samples:num_train_samples+num_val_samples+num_test_samples, :]
    
        train_S1 = np.expand_dims(train_S1, axis=-1)
        train_S1 = np.repeat(train_S1, 1, axis=-1)
        val_S1 = np.expand_dims(val_S1, axis=-1)
        val_S1 = np.repeat(val_S1, 1, axis=-1)
        test_S1 = np.expand_dims(test_S1, axis=-1)
        test_S1 = np.repeat(test_S1, 1, axis=-1)
    
    
        # Repeat and expand dimensions for S1 to make size 3
        train_S1 = np.expand_dims(train_S1, axis=-1)
        train_S1 = np.repeat(train_S1, 3, axis=-1)
        val_S1 = np.expand_dims(val_S1, axis=-1)
        val_S1 = np.repeat(val_S1, 3, axis=-1)
        test_S1 = np.expand_dims(test_S1, axis=-1)
        test_S1 = np.repeat(test_S1, 3, axis=-1)
    
        return train_S1, train_y, val_S1, val_y, test_S1, test_y
    
    train_S1, train_y, val_S1, val_y, test_S1, test_y = split(plv, y, 0.3, 0.2, 0.5)
    del S1, y  # Assuming S1 and y are defined earlier
    
    # Data Loader 
    # Dataloader is jarring asf so you need to have the sample dimension as the first, so i need to reshape it as follows:
    
    # Reshape train_S1 and train_y to have a shape of (n, 22, 2567, 240,3)
    train_S1 = np.transpose(train_S1, (2,0,1,3,4))  # Move the sample dimension to the first dimension
    val_S1 = np.transpose(val_S1, (2,0,1,3,4))
    test_S1 = np.transpose(test_S1, (2,0,1,3,4))  # Move the sample dimension to the first dimension
    
    train_loader = tf.data.Dataset.from_tensor_slices((train_S1, train_y))
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
        image_size = [22, 22]  # Input image size
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
        CFG.preset, num_classes=CFG.num_classes,input_shape=(22,22,3)
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
    
    return model, accuracies
    

# accuracies, model = makemodel(subject_data['S2'][:,:],1,20,0.3,0.2,0.5)

# %%  Define the subject numbers
subject_numbers = [] # 1, 2, 5, 9, 21, 31, 34, 39
# Loop through each subject number
for subject_number in subject_numbers:
    # Call the makemodel function to create the model
    model, accuracies = makemodel(subject_data[f'S{subject_number}'][:,:], 1, 10, 0.3, 0.2, 0.5)
    
# 39 - [0.51666667 0.6        0.58333333]
# 34 - [0.5875     0.50416667 0.48333333]
# 31 - 
# 21 - 
# 9 - 
# 5 - 
# 2 - 
# 1 - [0.5375     0.56666667 0.4625    ]

    