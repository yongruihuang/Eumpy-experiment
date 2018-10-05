# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:02:31 2018

@author: Yongrui Huang
"""

import numpy as np
import build_model
import sys
import keras

if __name__ == '__main__':
    
    #record training process file res.log
    #saveout = sys.stdout                                     
    #fsock = open('res.log', 'w')
    
    #load dataset set, in this way, the loading process will be very quick
    train_X = np.load("../../dataset/fer2013/train_X.npy")
    #print (X_train.shape)
    train_y = np.load("../../dataset/fer2013/train_y.npy")
    #print (y_train.shape)
    validation_X = np.load("../../dataset/fer2013/validation_X.npy")
    validation_y = np.load("../../dataset/fer2013/validation_y.npy")
    
    mean_X = np.load("../../dataset/fer2013/X_mean.npy")
    train_X -= mean_X
    train_X = train_X.reshape(train_X.shape[0], 48, 48, 1)
    train_y = keras.utils.to_categorical(train_y, num_classes = 7)
    
    validation_X -= mean_X
    validation_X = validation_X.reshape(validation_X.shape[0], 48, 48, 1)
    validation_y = keras.utils.to_categorical(validation_y, num_classes = 7)
    
    
    model = build_model.get_model()
   
    epochs = 16
    batch_size = 128
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=2, validation_data = (validation_X, validation_y))
    build_model.plot_training(history, "base")
    
    #fsock.close() 
    model.save('../../model/CNN_expression_baseline.h5')
    print ("finish")
    
     
