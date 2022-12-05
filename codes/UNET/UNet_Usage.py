from tensorflow import keras
from UNET.Code.data_utils import *
import shutil
import os
import cv2

            
def retrain_UNet(model_path:str,save_path:str,x_train_path:str , y_train_path:str , Epochs:int):
    """
      Retrain The Pretrained UNet that saved using new datas and replace the saved model with retrained model 

      Arguments:
         x_train_path -- train lung data path
         y_train_path -- train lung_mask data path
         Epochs -- number of epochs
    """   
    x_train = np.load(x_train_path)  # load x_train
    y_train = np.load(y_train_path)  # load y_train
    #load pretrained model
    model = keras.models.load_model(model_path, custom_objects = { "dice_coef": dice_coef})
    checkpoint = keras.callbacks.ModelCheckpoint(save_path+'{epoch:02d}.model', period = 10) 
    model.fit(x_train, y_train,callbacks=[checkpoint],epochs =Epochs, batch_size = 64, shuffle =  True)  #train model using new datas
    # shutil.rmtree('unet_seg_model/seg_model.model', ignore_errors=True)   #remove old model
    model.save(save_path+'.model')  #save new model

def save_predicted_masks(preds : np.ndarray , des_path : str):
    """
      Save predicted masks as black and white only and shape of (512,512) 

      Arguments:
         preds -- predicted lung_masks arrays
         des_path -- destination path to save generated lung_masks
    """   
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    if not os.path.isdir(des_path): # if the destination path doesnt exist make it
        os.mkdir(des_path)
    # make an empty arrat for processed lung_masks
    des_preds = np.empty((len(preds), IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32)
    index = 0
    for x in preds:
        x = cv2.resize(x, dsize=(IMG_WIDTH, IMG_HEIGHT)) #resize to (512,512)
        x = np.expand_dims(x, axis = 2) #(512,512,1)
        x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB) #convert to RGB (512,512,3)
        # convert lung_masks to black and white
#         for i, item in enumerate(x):
#             for j, item1 in enumerate(x[i]):
#                 for z, item2 in enumerate(x[i][j]):
#                     if(x[i][j][z] < 0.3):
#                         x[i][j][z]=0
#                     if(x[i][j][z] >= 0.3):
#                         x[i][j][z]=1
        x [x>=x.mean()]=1
        x [x<x.mean()]=0
        
        des_preds[index] = x
        index += 1
    print(des_preds.shape)
    np.save(des_path+'pred_masks.npy', des_preds)

def generate_Lung_Masks_Using_UNet(model_path,x_path, des_path):
    """
      Generate lung_msks for unlabeled images using pretrained UNet

      Arguments:
         x_path -- unlabeled images path
         des_path -- destination path to save generated lung_masks
    """   
    X = np.load(x_path) #load unlabeled data
    model = keras.models.load_model(model_path, custom_objects = { "dice_coef": dice_coef}) #load pretrained model
    preds = model.predict(X) #predict lung_masks
    save_predicted_masks(preds, des_path) #process and save the predicted lun_masks