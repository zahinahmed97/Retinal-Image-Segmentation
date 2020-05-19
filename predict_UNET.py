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

import sys

from matplotlib import pyplot as plt
from matplotlib import figure
#Keras
from keras.models import model_from_json
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

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
N_epochs = 150
batch_size = 32


#Choose the model to test: best==epoch with min loss, last==last epoch
best_last = 'best'

#number of full images for the test (max 20)
full_images_to_test = 20

#How many original-groundTruth-prediction images are visualized in each image
N_group_visual = 1

#Compute average in the prediction, improve results but require more patches to be predicted
average_mode = True

#Only if average_mode==True. Stride for patch extraction, lower value require more patches to be predicted
stride_height = 5
stride_width = 5


test_imgs_orig = load_hdf5("DRIVE_datasets_training_testing\\DRIVE_dataset_imgs_test.hdf5")
test_imgs_gtruth=load_hdf5("DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5")
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

#visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'imgs_test').show()
#visualize(group_images(test_imgs_gtruth[0:20,:,:,:],5),'test_gtruth').show()

#the border masks provided by the DRIVE
test_border_masks = load_hdf5("DRIVE_datasets_training_testing\\DRIVE_dataset_borderMasks_test.hdf5")

assert (stride_height < patch_height and stride_width < patch_width)

Imgs_to_test =full_images_to_test

#Grouping of the predicted images
N_visual = N_group_visual 

patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None

patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(DRIVE_test_imgs_original = "DRIVE_datasets_training_testing\\DRIVE_dataset_imgs_test.hdf5",#original
DRIVE_test_groudTruth ="DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5",  #masks
Imgs_to_test = Imgs_to_test,
patch_height = patch_height,
patch_width = patch_width,
stride_height = stride_height,
stride_width = stride_width
)

#Load the saved model
model = model_from_json(open('model_architecture_trial2.json').read())
model.load_weights('Model_best_weights_trial2.h5')


predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print ("predicted images size :")
print (predictions.shape)


pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")


pred_imgs = None
orig_imgs = None
gtruth_masks = None

pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
gtruth_masks = masks_test  #ground truth masks

# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization

## back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print ("Orig imgs shape: " +str(orig_imgs.shape))
print ("pred imgs shape: " +str(pred_imgs.shape))
print ("Gtruth imgs shape: " +str(gtruth_masks.shape))
visualize(group_images(orig_imgs,N_visual),"all_originals_trial2")#.show()
visualize(group_images(pred_imgs,N_visual),"all_predictions_trial2")#.show()
visualize(group_images(gtruth_masks,N_visual),"all_groundTruths_trial2")#.show()

assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(total_img,"_Original_GroundTruth_Prediction_trial2"+str(i))#.show()

#====== Evaluate the results
print ("\n\n========  Evaluate the results =======================")
#predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV
print ("Calculating results only inside the FOV:")
print ("y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)")
print ("y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)")

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print ("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig("ROC_trial2.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig("Precision_recall_trial2.png")


#Confusion matrix
threshold_confusion = 0.5
print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print ("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print ("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure): " +str(F1_score))

#Save the results
file_perf = open('performances_trial2.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()
