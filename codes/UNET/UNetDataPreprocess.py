import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split



def read_images_masks(image_path : str ,mask_path :str , base = "none") -> tuple():
    """
      Extract all of images and masks in given directory

      Arguments:
         image_path -- path to all lung images
         mask_path -- path to lung masks
      Returns:
          tuple of images an masks list that contain path to images and masks
           
    """    
    directory_contents = os.listdir(image_path) # extract all contents in the path
    directory_contents2 = os.listdir(mask_path)
    images = []
    # first of the name of images to make clear which database image it belongs: for instance 'coronacases'
    bases = []
    masks = []
    
    # extract all base names
    
    if(base != "none"):
        bases.append(base)
    else:
        for i in directory_contents:
            name = i.split('.')[0].split('_')[0]
            if(name not in bases):
                bases.append(name)
    # for each base find images and masks             
    for base in bases:
        print(base)
        for i in directory_contents:
            name = i.split('.')[0].split('_')[0]
            if(name == base):
                images.append(image_path + i)
        images.sort()
        print("len images : ",len(images))
    
        for j in directory_contents2:
            name = i.split('.')[0].split('_')[0]
            if(name == base):
                masks.append(mask_path + j)
    
        masks.sort()
        print("len masks : ",len(masks))
    
    return images , masks 

def save_train_test_data_for_u_net(dataX_64 : np.ndarray ,dataY_64 : np.ndarray, dataX_512 : np.ndarray,dataY_512 : np.ndarray
                                   ,save_dir:str ,test_size = 0.2):
    
    """
      Save dataset as train and test data for UNet training as (64,64) & (512,512)

      Arguments:
         dataX_64 -- array of lung images with size of (64,64)
         dataY_64 -- array of lung_masks with size of (64,64)
         dataX_512 -- array of lung images with size of (512,512)
         dataY_512 -- array of lung_masks with size of (512,512)
         save_dir -- path to save the dataset
         test_size -- percentage of test data when split dataset
          
    """
    # split dataset to train and test
    trainX_64, testX_64 = train_test_split(dataX_64, test_size = test_size, random_state=50)
    trainY_64, testY_64 = train_test_split(dataY_64, test_size = test_size, random_state=50)
    
    trainX_512, testX_512 = train_test_split(dataX_512, test_size = test_size, random_state=50)
    trainY_512, testY_512 = train_test_split(dataY_512, test_size = test_size, random_state=50)


    if not os.path.exists(save_dir): # if there is no exist, make the path
        os.makedirs(save_dir)
    
    #save dataset
    np.save(save_dir+'x_train_64.npy', trainX_64)
    np.save(save_dir+'y_train_64.npy', trainY_64)
    np.save(save_dir+'x_test_64.npy', testX_64)
    np.save(save_dir+'y_test_64.npy', testY_64)
    np.save(save_dir+'x_train_512.npy', trainX_512)
    np.save(save_dir+'x_test_512.npy', testX_512)
    np.save(save_dir+'y_train_512.npy', trainY_512)
    np.save(save_dir+'y_test_512.npy', testY_512)
    
    print(trainX_64.shape, trainY_64.shape)
    print(testX_64.shape, testY_64.shape)
    print(trainX_512.shape, trainY_512.shape)
    print(testX_512.shape, testY_512.shape)
    

def normalize(img : np.ndarray) -> np.ndarray:
    """
      Normalize image array 

      Arguments:
         img -- image array that need to be normalize
      Returns:
         img -- noraml image array
          
    """    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def preprocess_lungs(images : list) -> tuple:
    """
      Preprocess lung images to use in UNet that should be (64,64,1) and also as (512,512,1) for later usage

      Arguments:
         images -- images list of paths:str
      Returns:
         (dataX_64,dataX_512) -- tuple of np.ndarrays all preprocessed images
          
    """    
    dataX_64 = np.empty((len(images), 64, 64, 1), dtype=np.float32)  # make an empty array of (len(images)*64*64*1)
    dataX_512 = np.empty((len(images), 512, 512, 1), dtype=np.float32)  # make an empty array of (len(images)*512*512*1)
    index = 0
    for i in images:
        img = cv2.imread(i, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32') # read image 
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_512 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to 2D image
        img = cv2.resize(img_512, dsize=(64, 64)) #resize to (64,64)
        # normalize image
        img =  normalize(img) 
        img_512 =  normalize(img_512)
        #add image to the array
       
        dataX_64[index] = np.expand_dims(img, axis = 2) 
        dataX_512[index] = np.expand_dims(img_512, axis = 2)
        
        index = index + 1
    return dataX_64,dataX_512
    

def preprocess_lung_masks(lung_masks : list) -> tuple:
    """
      Preprocess lung_masks to use in UNet that should be (64,64,1) and also as (512,512,1) for later usage

      Arguments:
         lung_masks -- lung_masks list of paths:str
      Returns:
         (dataY_64,dataY_512) -- tuple of np.ndarrays all preprocessed lung_masks
          
    """   
    dataY_64 = np.empty((len(lung_masks), 64, 64, 1), dtype=np.float32) # make an empty array of (len(lung_masks)*64*64*1)
    dataY_512 = np.empty((len(lung_masks), 512, 512, 1), dtype=np.float32) # make an empty array of (len(lung_masks)*512*512*1)
    index = 0
    for mask_path in lung_masks: 
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32') #read image
        mask_512 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # convert to 2D image
        mask = cv2.resize(mask_512, dsize=(64, 64)) #resize to (64*64)
        #add mask to array
        dataY_64[index] = np.expand_dims(mask, axis = 2) 
        dataY_512[index] = np.expand_dims(mask_512, axis = 2)
        index = index + 1
    # normalize mask
    dataY_64 /= 255.    
    dataY_512 /= 255.   
    
    return dataY_64 , dataY_512

def preprocess_lungs_for_UNet_training(image_path : str, mask_path:str ,maxNumOfLungs :int,save_dir:str):
    """
      Preprocess lungs and lung_masks to use in UNet training using above functions

      Arguments:
         image_path -- all lung images path 
         mask_path -- all lung_masks path
         maxNumOfLungs -- number of images that can be used for training
         save_dir --  path to save the preprocessed data
    """   
    # inititalize images and masks list
    images , lung_masks  = read_images_masks(image_path,mask_path)
    # preprocess lungs 
    if(maxNumOfLungs == 0):
        maxNumOfLungs = len(images) -1 
    dataX,dataX_512 = preprocess_lungs(images[:maxNumOfLungs])
    # preprocess lung_msaks
    dataY , dataY_512 = preprocess_lung_masks(lung_masks[:maxNumOfLungs])
    # save lungs and lung_masks as train and test dataset
    save_train_test_data_for_u_net(dataX ,dataY ,dataX_512 ,dataY_512 ,save_dir)
    

def preprocess_lungs_for_UNet_prediction(image_path:str, save_dir:str):
    """
      Preprocess lungs to use in UNet prediction 

      Arguments:
         image_path -- all lung images path 
         save_dir --  path to save the preprocessed data
    """   
    images = []
    # extract all contents in the path
    directory_contents = os.listdir(image_path)
    for i in directory_contents:
        # add each image path in the images path to the images list
        images.append(image_path + i)
        images.sort()
    print("len images : ",len(images))
    
    dataX,dataX_512 = preprocess_lungs(images)   #preprocess lungs
    if not os.path.exists(save_dir): # if there is no exist, make the path
        os.makedirs(save_dir) 
    # save images 
    np.save(save_dir+'x_64.npy', dataX) 
    np.save(save_dir+'x_512.npy', dataX_512)