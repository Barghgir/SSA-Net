import numpy as np
import nibabel as nib
import os
import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler



def load_data_nii(file_path: str) -> np.ndarray:
    """
    Get the array of nii file

    Arguments:
        file_path -- Address of nii file
    Returns:
        arr --  Numpy array of nii file
    """

    data = nib.load(file_path)
    arr = data.get_data()
    return arr



def normalizer(arr: np.ndarray) -> np.ndarray:
    """
    Normalize each slice of image

    Arguments:
        arr -- arrays of each image slices
    Returns:
        norm --  Normalized array that limited in [0, 255] 
    """
    # norm = normalize(arr, norm='max', axis=0)

    # scaler = MinMaxScaler(feature_range=(0, 255), copy=True, clip=False)
    # norm = scaler.fit_transform(arr)
    # print(norm.shape)
    # print(norm)

    X_std = (arr - arr.min(axis=None)) / (arr.max(axis=None) - arr.min(axis=None))
    norm = X_std * (255 - 0)
    return norm


def crop(arr: np.ndarray) -> np.ndarray:
    """
    Crop the image with center of it to recieve (512, 512, :) shape 

    Arguments:
        arr -- Arrays of each image slices
    Returns:
        crop_img --  Cropted image
    """

    center = arr.shape[0]/2
    pos = center + 256
    neg = center - 256
    crop_img = arr[int(neg):int(pos), int(neg):int(pos)]
    # img = np.reshape(crop_img, (1, crop_img.shape[0], crop_img.shape[1]))
    img = crop_img
    return img


def build_path(folder_path: str):
    """
    This function bulds the input path and if it wasn't exist

    Arguments:
        folder_path -- The path you want to built
    """

    new_path_1 = folder_path
    file_path = ""
    for folder in new_path_1.split("/"):
        file_path += folder + "/"
        if not os.path.isdir(file_path) and file_path != "":
            os.mkdir(file_path)
            print(f'{file_path} is beild...')
    
    return file_path



def mixer(output_path: str, dataframe: pd.DataFrame, image_col: str,  mask_col: str):
    """
    Mix (actually multiply) mask and image with dataframe that we give to function  

    Arguments:
        output_path -- Where you want to save mixed images
        dataframe -- Dataframe of all the image and masks with PNG format
        image_col -- Column of Image
        mask_col -- Column of Mask

    """
    build_path(folder_path=output_path)
    for i in tqdm(range(len(dataframe))):
        image = plt.imread(dataframe[image_col][i])
        mask = plt.imread(dataframe[mask_col][i])

        mask[mask > mask.mean()] = 1
        mask[mask <= mask.mean()] = 0

        mixed_img = mask*image

        if not output_path.endswith("/"):
            output_path += "/"

        address = dataframe[image_col][i].split('.')[0]
        file_path = address.split("/")[-1]
        # cv2.imwrite(f"{output_path}{file_path}.png", mixed_img)
        plt.imsave(f"{output_path}{file_path}.png", mixed_img)  


def get_ssa_input(inf_mask_path: str, lung_mask_addr:str, output_path:str):
    """
    get input of SSA_Net model and save it with .npz file
    Arguments:
        inf_mask_path --> Path of .nii.gz Inflation mask file with skape (512, 512, number of slices)
        lung_mask_addr --> Path of png Lung mask FOLDER
        output_path --> Path of Folder that we want to save

    """
    # work with paths
    build_path(output_path)

    if not lung_mask_addr.endswith("/"):
        lung_mask_addr += "/"
    if not output_path.endswith("/"):
        output_path += "/"

    # Inflation masks
    inf_mask = load_data_nii(file_path = inf_mask_path)
    inf_mask = crop(inf_mask)
    inf_mask[inf_mask == 3] = 0

    lung_mask_lst = glob.glob(lung_mask_addr+"*.png")
    lung_mask_lst.sort()
    new_mask = np.zeros(shape=(inf_mask.shape[0], inf_mask.shape[1], 3))
    
    for i, file in enumerate(lung_mask_lst):
        # Lung mask
        lung_mask = cv2.imread(file, 0)
        
        # plt.imshow(inf_mask[:,:,i], cmap='gray')
        # plt.show()

        lung_mask[lung_mask == 255] = 1
        lung_mask = lung_mask.T
        # print("inf_mask   Number of 3 ---> ", np.sum(inf_mask == 3))
        new_mask = np.zeros(shape=(inf_mask.shape[0], inf_mask.shape[1], 3))
        
        new_mask[:, :, 0] = lung_mask
        new_mask[:, :, 1][inf_mask[:,:,i] == 1] = 1
        new_mask[:, :, 2][inf_mask[:,:,i] == 2] = 1

        file_name = inf_mask_path.split("/")[-1]
        file_name = file_name.split(".")[0]
        
        np.savez(f"{output_path}{file_name}{i:04}.npz", new_mask)


def turn2png(img_path: str, output_path: str, is_mask: bool):
    
    """
    Convert the .nii.gz files to png slices 

    Arguments:
        img_path -- Folder path of .nii.gz files.
        output_path -- Path of folder that you want to save them.
        is_mask -- when gets true that we use mask

    """

    # this make output path
    new_path = build_path(folder_path = output_path)
    
    # this checks img_path ends with "/"
    if not img_path.endswith("/"):
        img_path += "/"

    # this find files that ends with "*.nii.gz"
    for file in glob.glob(img_path+"*.nii.gz"):
        
        f = file.split("/")[-1]
        img = load_data_nii(file_path=img_path+f)
        print(f," ---------> ",img.shape)

        for i in range(img.shape[-1]): 
            cropped_img = crop(arr=img[:, :, i])
            if not is_mask:
                norm_img = normalizer(arr=cropped_img)
            else:
                norm_img = cropped_img

            im = norm_img.T
            # cv2.imwrite(f"{new_path}{f.split('.')[0]}_{i:04}.png", im)
            plt.imsave(f"{new_path}{f.split('.')[0]}_{i:04}.png", im, cmap='gray')


def turn2png_2(img_path: str, output_path: str, is_mask: bool):
    
    """
    Convert the .nii.gz files to png slices. This is for DICOM part

    Arguments:
        img_path -- Folder path of .nii.gz files.
        output_path -- Path of folder that you want to save them.
        is_mask -- when gets true that we use mask

    """

    # this make output path
    new_path = build_path(folder_path = output_path)
    
    # this checks img_path ends with "/"
    if not img_path.endswith("/"):
        img_path += "/"

    # this find files that ends with "*.nii.gz"
    for file in glob.glob(img_path+"*.nii.gz"):
        
        f = file.split("/")[-1]
        img = load_data_nii(file_path=img_path+f)
        print(f," ---------> ",img.shape)

        for i in range(img.shape[-1]-1, -1, -1): 
            cropped_img = crop(arr=img[:, :, i])
            if not is_mask:
                norm_img = normalizer(arr=cropped_img)
            else:
                norm_img = cropped_img

            im = norm_img.T
            j = img.shape[-1] - i - 1
            # cv2.imwrite(f"{new_path}{f.split('.')[0]}_{i:04}.png", im)
            plt.imsave(f"{new_path}{f.split('.')[0]}_{j:04}.png", im, cmap='gray')




