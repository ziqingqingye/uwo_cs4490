import cv2

import numpy as np
import os

import keras.backend as K
K.set_image_data_format('channels_first')
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_DATA = ['1.png','2.png','3.png']

i_size = 256

classes = [0. ,  1.,  2.,   3.  , 4.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 



    
def inference(stride,model='segnet.ckpt',test_data_dir='../data/all_test/test/',output_path='../predict/pre'):
  
    model = load_model(model) # load the model

    # read all test data
    for k in range(len(TEST_DATA)):
        path = TEST_DATA[k]
        image = cv2.imread(test_data_dir + path)
        
        ph = (image.shape[0]//stride + 1) * stride  # calculate the size of image according the stride
        pw = (image.shape[1]//stride + 1) * stride
        pimg = np.zeros((ph,pw,3),dtype=np.uint8)
        pimg[0:image.shape[0],0:image.shape[1],:] = image[:,:,:]
    
        pimg = img_to_array(pimg.astype("float") / 255.0)
        print('src:',pimg.shape)
        mask = np.zeros((ph,pw),dtype=np.uint8) # define a mask array
        for i in range(ph//stride):
            for j in range(pw//stride):
                block = pimg[:3,i*stride:i*stride+i_size,j*stride:j*stride+i_size] # crop a sub image
                if block.shape[1] == 256 and block.shape[2] == 256:


                    pred = model.inference_classes(np.expand_dims(block, axis=0),verbose=2) # predict the label
                    pred = labelencoder.inverse_transform(pred[0]).reshape((256,256)).astype(np.uint8) # transform one hot code to normal label
                    mask[i*stride:i*stride+i_size,j*stride:j*stride+i_size] = pred[:,:] # save the result to mask array
        cv2.imwrite(output_path+{}+'.png'.format(k+1),mask[0:image.shape[0],0:image.shape[1]]) # save the mask to disk

inference(i_size)



