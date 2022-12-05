from keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from sklearn import model_selection, metrics
from sklearn.metrics import classification_report
import math

from tensorflow import keras

import cv2

# requirements -> hausdorff , numba , monai

def dice_Similarity_coef(G, S):
    """
    Argumants :
        G -- Ground truth
        S -- Segmentation result
    """
    G = K.flatten(G)
    S = K.flatten(S)
    
    intersection = K.sum(G * S)
    return (2. * intersection + K.epsilon()) / (K.sum(G) + K.sum(S) + K.epsilon())


from hausdorff import hausdorff_distance
def hausdoff_distance_func(G, S):
    """
    Argumants :
        G -- Ground truth
        S -- Segmentation result
    """
    hd_lst = [hausdorff_distance(G[i, :,:,j].numpy(), S[i, :,:,j].numpy(), distance='euclidean') for j in range(G.shape[-1]) for i in range(G.shape[0])]
    hd = sum(hd_lst) / len(hd_lst)
    return hd

def mean_absolute_error(G, S):
    """
    Argumants :
        G -- Ground truth
        S -- Segmentation result
    """
    H = 512
    W = 512
    G = K.flatten(G)
    S = K.flatten(S)
    return (1/(H*W))*(K.sum(G) + K.sum(S) + K.epsilon())

def custom_loss(G,s):
    S,y1,y2,y3,y4 = s
    seg_loss = dice_Similarity_coef(G,S)
    at_loss1 = mean_absolute_error(y1,y2)
    at_loss2 = mean_absolute_error(y2,y3)
    at_loss3 = mean_absolute_error(y3,y4)
    losses =[float(at_loss1.numpy()) , float(at_loss2.numpy()) , float(at_loss3.numpy())]
    sa_loss = sum(losses)/len(losses)
    
    alpha = 0.1
    
    sum_loss = seg_loss + alpha * sa_loss
    
    return sum_loss

