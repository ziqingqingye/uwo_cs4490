#coding=utf-8

import cv2
import random
import os
import numpy as np
# this module is used to enrich image data sets,in order to increase the robustness of the model

weight = 256  
height = 256  

datas = ['1.png','2.png','3.png','4.png','5.png']


    
# use opencv to rotate image
def img_rotate(img,label,angle):
    M_img_rotate = cv2.getRotationMatrix2D((weight/2, height/2), angle, 1)
    img = cv2.warpAffine(img, M_img_rotate, (weight, height))
    label = cv2.warpAffine(label, M_img_rotate, (weight, height))
    return img,label

# use opencv to blur image
def img_blur(img):
    img = cv2.img_blur(img, (3, 3));
    return img

# add noise
def add_noise(img):
    for i in range(200):
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img
    
# augment image data randomly
def augment(img,label):
    if np.random.random() < 0.25:
        img,label = img_rotate(img,label,90)
    if random.random() < 0.25:
        img,label = img_rotate(img,label,180)
    if random.random() < 0.25:
        img,label = img_rotate(img,label,270) # 25% chance to rotate image
    if random.random() < 0.25:
        img = cv2.flip(img, 1)  
        label = cv2.flip(label, 1) # 25% chance to flip image
        
 
        
    if np.random.random() < 0.25:
        img = img_blur(img) # 25% chance to blur image
    
    if np.random.random() < 0.2:
        img = add_noise(img) # 25% chance to add noise image
        
    return img,label

def create(input='./data',output='./aug/train',mode = None):

  
    gc = 0
    for i in range(len(datas)):
        # read image and labels

        
        source = cv2.imread(input+'/src/' + datas[i])  # 3 channels
        target = cv2.imread(input+'/label/' + datas[i],cv2.IMREAD_GRAYSCALE)  # single channel
        X_height,X_width,_ = source.shape
        for j in range(10000):
            # crop the image and label randomly
            rw = random.randint(0, X_width - weight - 1)
            rh = random.randint(0, X_height - height - 1)

            img_block = source[rh: rh + height, rw: rw + weight,:]
            label_block = target[rh: rh + height, rw: rw + weight]
            if mode == 'aug':
                # enhance the dataset
                img_block,label_block = augment(img_block,label_block)

            vs = label_block *50
            # create the output dir
            if os.path.exists(output+'/vs/')==False:
                os.makedirs(output+'/vs/')
            if os.path.exists(output+'/src/')==False:
                os.makedirs(output+'/src/')
            if os.path.exists(output+'/label/')==False:
                os.makedirs(output+'/label/')
            # save  result
            
            cv2.imwrite((output+'/vs/%d.png' % gc),vs)
            cv2.imwrite((output+'/src/%d.png' % gc),img_block)
            cv2.imwrite((output+'/label/%d.png' % gc),label_block)

            gc += 1


            
    


create(mode='aug')
