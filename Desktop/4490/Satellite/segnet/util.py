import cv2
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
classes = [0., 1., 2., 3., 4.]

# encode the label to onehot
labelencoder = LabelEncoder()
labelencoder.fit(classes)

# read the image from path,we can specify its mode
def loadimg(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img

# the function is used to draw training result
def draw(epoch,f,savepath):
    plt.style.use("ggplot")
    plt.figure() # create a figure
    r = np.arange(0, epoch) # the range of train epoch
    loss = f.history["loss"] # the loss of each epoch
    val_loss = f.history['val_loss'] # the validation loss of each epoch
    acc = f.history["acc"] # the accuracy of each epoch
    val_acc = f.history["val_acc"] # the validation accuracy of each epoch
    plt.plot(r, loss, label="train loss") # plot train loss
    plt.plot(r, val_loss, label="validation loss") # plot validation loss
    plt.plot(r, acc, label="train accuracy") # plot train accuracy
    plt.plot(r, val_acc, label="val accuracy") # plot validation accuracy
    plt.title("Training Loss and Accuracy") # plot title
    plt.xlabel("Epoch #") # plot xlabel
    plt.ylabel("Loss/Accuracy") # plot ylabel
    plt.legend(loc="lower left") # plot legend
    plt.savefig(savepath) # save the figure

# the function is used to generate data for training or validation,use yield key word in the loop to make it to be a generator
def generateData(data_root,batch_size,weight,height,classes_num, mode='train', vrate=0.25):
    # print 'generateData...'
    total = [] # used to store all image path

    for pic in os.listdir(data_root+ 'src'):
        total.append(pic)  # load all pathes of data
    random.shuffle(total) # shuffle the data
    total_num = len(total) # the the total number of data
    val_num = int(vrate * total_num) # the the number of validation data accourding the validation rate
    data = total[-1 * val_num:] if mode == 'train' else total[val_num:] # get the data accoording the data  parameter

    while True:
        train_data = []
        traiclasses_num = []
        batch = 0
        for i in (range(len(data))): # traverse  all datas
            url = data[i]
            batch += 1
            img = loadimg(data_root + 'src/' + url) # read the image using RGB mode
            img = img_to_array(img)
            train_data.append(img)
            label = loadimg(data_root + 'label/' + url, grayscale=True) # read the label
            label = img_to_array(label).reshape((weight * height,)) # reshape the label
            # print label.shape
            traiclasses_num.append(label)
            if batch % batch_size == 0: # collection the data batch to the multiply of batch_size
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                traiclasses_num = np.array(traiclasses_num).flatten()
                traiclasses_num = labelencoder.transform(traiclasses_num) # transform the label to one hot using label encoder
                traiclasses_num = to_categorical(traiclasses_num, num_classes=classes_num)
                traiclasses_num = traiclasses_num.reshape((batch_size, weight * height, classes_num)) # reshape the 2D label to 1D
                yield (train_data, traiclasses_num) # generate the data
                train_data = []
                traiclasses_num = []
                batch = 0

