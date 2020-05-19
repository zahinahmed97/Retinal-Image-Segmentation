# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:53:25 2020

@author: Zahin Ahmed
"""

import numpy as np
from PIL import Image
import cv2
import random

from help_functions import *
from pre_processing import *
from extract_patches import *

from keras.models import Model
from keras import losses
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *

import sys

from matplotlib import pyplot as plt
from matplotlib import figure

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

patch_height = 48
patch_width = 48
#number of total patches:
N_subimgs = 190000
#if patches are extracted only inside the field of view:
inside_FOV = False
#Number of training epochs
N_epochs = 300
batch_size = 32


   
train_imgs_original = load_hdf5("DRIVE_datasets_training_testing\\DRIVE_dataset_imgs_train.hdf5")
train_masks = load_hdf5("DRIVE_datasets_training_testing\DRIVE_dataset_groundTruth_train.hdf5") #masks always the same

#visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train').show()
#visualize(group_images(train_masks[0:20,:,:,:],5),'ground_truth_train').show()

train_imgs=my_PreProc(train_imgs_original)

#visualize(group_images(train_imgs[0:20,:,:,:],5),'processed_imgs_trial2').show()

train_masks = train_masks/255.

train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
data_consistency_check(train_imgs,train_masks)

assert(np.min(train_masks)==0 and np.max(train_masks)==1)

print ("\ntrain images/masks shape:")
print (train_imgs.shape)
print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
print ("train masks are within 0-1\n")

#extract the TRAINING patches from the full images
patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
data_consistency_check(patches_imgs_train, patches_masks_train)

print ("\ntrain PATCHES images/masks shape:")
print (patches_imgs_train.shape)
print ("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))





#Define the neural network
def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


    
#======== Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+"sample_input_imgs_trial3")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+"sample_input_masks_trial3")#.show()

n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model



print ("Check: final output of the network:")
print (model.output_shape)
model.summary()
plot(model, to_file='./'+ 'model.png')
json_string = model.to_json()
open('./'+'model_architecture_trial3.json', 'w').write(json_string)

#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+'Model_best_weights_trial2.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased



patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size,initial_epoch=150, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

