#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import keras.backend as K
from model import Network
K.set_image_data_format('channels_first')
from util import *

from keras.callbacks import ModelCheckpoint

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # define the GPU number
epoch = 30 # define the train epoch
batch_size = 50 # define the batch size
weight = 256 # define the image size
height = 256
classes_num = 5 # define the number of classes
image_sets = ['1.png','2.png','3.png']
data_root ='../data/seg_train/train/' # specify the data root path

def train(model_path,plot):

    model = Network(weight,height,classes_num) # create a model

    # train the model
    f = model.fit_generator(generator=generateData(data_root,batch_size,'train',0.25,weight,height,classes_num),epochs=epoch,verbose=1,steps_per_epoch=10,validation_steps=10,
                    validation_data=generateData(data_root,batch_size,'val',0.25,weight,height,classes_num),callbacks=[ModelCheckpoint(model_path,monitor='val_acc',save_best_only=True,mode='max')  ])

    # draw the train result
    draw(epoch,f,plot)





model_path='segnet.ckpt' #define the model saved path
plot='plot.png' #define the plot path
train(model_path,plot) #train
    
