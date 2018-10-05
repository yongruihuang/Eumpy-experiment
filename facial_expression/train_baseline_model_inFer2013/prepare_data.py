# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 08:53:25 2018

@author: Yongrui Huang
"""

import pandas as pd
import numpy as np
from numpy import uint8
import matplotlib.pyplot as plt
from sklearn.model_selection._split import train_test_split

#origin: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
#new_type: 0=Happy, 1=Sad, 2=Surprise, 3=Neutral
def show_demon_image(img):
    '''
    img: numpy array respresent an image
    '''
    
    plt.figure("demon")
    plt.imshow(img)
    #plt.axis('off')
    plt.show()

def load_data(extend_disgust):
    '''
    extract data from 'fer2013.csv' file
    
    extend_digust: whether to extend disgust class
    
    return: numpy array -like
        train_X:       shape(?,48,48)
        validation_X:  shape(?,48,48) 
        train_y:       shape(?, )
        validation_y:  shape(?, )
    '''
    
    data = pd.read_csv("../../dataset/fer2013/fer2013.csv")
    
    X = []
    y = []
    for (pixels, emotion) in zip(data['pixels'], data['emotion']):
        #if emotion == 0 or emotion == 1 or emotion == 2:
        #   continue
        img = np.array((pixels.split(' ')), dtype=uint8 )
        img = img.reshape((48, 48))
        #img = cv2.equalizeHist(img)
        y.append(emotion)
        X.append(img)
    
    if extend_disgust:
        #extend disgust facial expression data, inorder to overcome the problem that class 'digust' has much less sample than other class.
        disgust_image = np.load('../../dataset/fer2013/extend_disgust.npy')
        X.extend(disgust_image)
        y.extend(np.ones((len(disgust_image),)))
    
    X = np.array(X, dtype=uint8)
    y = np.array(y, dtype=uint8)
    X = X.astype('float32')
    train_X, validation_X, train_y, validation_y = \
    train_test_split(X, y, test_size=0.2, random_state = 0)
    
    return train_X, validation_X, train_y, validation_y


if __name__ == '__main__':
    train_X, validation_X, train_y, validation_y = load_data(extend_disgust = True)
    
    #save data for quicker loading
    np.save("../../dataset/fer2013/train_X.npy",train_X)
    np.save("../../dataset/fer2013/train_y.npy",train_y)
    np.save("../../dataset/fer2013/validation_X.npy",validation_X)
    np.save("../../dataset/fer2013/validation_y.npy",validation_y)
    
    #save mean for normalization
    X_mean = np.mean(train_X, axis = 0)
    np.save("../../dataset/fer2013/X_mean.npy", X_mean)
    
    
    