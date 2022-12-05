import random 
from sklearn.utils import resample
from PIL import Image
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def upsample(a , b):
  a_upsampled = resample(a,replace=True, n_samples=len(b), random_state=42)
  return a_upsampled

def batch_upsample(train_dataset):
  numpy_dataset = np.stack(list(train_dataset))

  class_a = []
  class_b = []
  class_c = []
  input = numpy_dataset[0][0]
  target = numpy_dataset[0][1]
  index = 0
  for i in target:
    if((np.unique(i[:,:,1]).any()==0) == False and (np.unique(i[:,:,2]).any()==0) == False):
      class_a.append((input[index],i))
    if((np.unique(i[:,:,1]).any()==0) == False and (np.unique(i[:,:,2]).any()==0) == True):
      class_b.append((input[index],i))
    if((np.unique(i[:,:,1]).any()==0) == True and (np.unique(i[:,:,2]).any()==0) == False):
      class_c.append((input[index],i))
    index += 1

  if(len(class_b) > len(class_c)):
    if((len(class_c) - len(class_b)) > 2):
      class_c_upsampled = upsample(class_c , class_b)
      class_c = class_c_upsampled
  elif(len(class_b) < len(class_c)):
    if((len(class_b) - len(class_c)) > 2):
      class_b_upsampled = upsample(class_b , class_c)
      class_b = class_b_upsampled
  return class_a , class_b , class_c


def concatenate_shuffle_lists(list_1,list_2,list_3=None):
  if(list_3 != None):
    temp = list_1 + list_2
    cnct = temp + list_3
  else:
    cnct = list_1 + list_2
  random.shuffle(cnct)
  return cnct

def tensor_to_image_array(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    img = Image.fromarray(tensor)
    img = np.asarray(img)
    return img
def flip_data(data):
  augmented_image = []
  # print(data[0].shape)
  flip_1_img = tf.image.random_flip_up_down(data[0])
  flip_1_msk = tf.image.random_flip_up_down(data[1])

  flip_2_img = tf.image.random_flip_left_right(data[0])
  flip_2_msk = tf.image.random_flip_left_right(data[1])

  augmented_image.append((flip_1_img,flip_1_msk))
  augmented_image.append((flip_2_img,flip_2_msk))

  return augmented_image

def find_max(path):
  dir_list = os.listdir(path)
  temp_list = []
  maximum = 0
  if((len(dir_list) == 0)==False):
    for i in dir_list:
      tmp = i.split('/')[-1].split('.')[0]
      tmp = int(tmp)
      temp_list.append(tmp)
    maximum = max(temp_list)
  return maximum

def save_augmented_batch(dataset,path):
  if not os.path.isdir(path):
    os.mkdir(path)
  images_path = path + "images/"
  if not os.path.isdir(images_path):
    os.mkdir(images_path)
  masks_path = path + "masks/"
  if not os.path.isdir(masks_path):
    os.mkdir(masks_path)

  index = find_max(images_path) + 1
  for i in dataset:
    plt.imsave(f"{path}images/{str(index)}.png",tensor_to_image_array(i[0]))  
    # plt.imsave(f"{path}masks/{str(index)}.png", tensor_to_image_array(i[1])) 
    np.savez(f"{path}masks/{str(index)}.npz",tensor_to_image_array(i[1]))
    index += 1 

  augmented_batch_images_paths = glob.glob('/content/Datasets/augmented_batch/images/*.png')
  augmented_batch_masks_paths = glob.glob('/content/Datasets/augmented_batch/masks/*.npz')
  augmented_batch_images_paths.sort()
  augmented_batch_masks_paths.sort()


  dict_path = {
      'mixed_img':augmented_batch_images_paths,
      'inf_mask_paths':augmented_batch_masks_paths,
  }
  data_paths2 = pd.DataFrame(dict_path)
  # data_paths.head(5)
  old_df = pd.read_csv('paths_lst.csv')
  old_df = old_df.append(data_paths2)
  old_df.to_csv('paths_lst.csv')

def batch_augmenter(train_dataset):
  class_a , class_b , class_c = batch_upsample(train_dataset)
  dataset = concatenate_shuffle_lists(class_a , class_b , class_c)
  random.seed(5)
  p = random.uniform(0, 1)
  augmented_images = []
  for i in dataset:
    if(p > 0.8):
      augmented_image = flip_data(i)
      augmented_images= augmented_images + augmented_image
  dataset = concatenate_shuffle_lists(dataset , augmented_images)
  path = "/content/Datasets/augmented_batch/"
  save_augmented_batch(dataset,path)





