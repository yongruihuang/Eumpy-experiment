# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:05:08 2018

@author: Yongrui Huang
"""

import keras.layers as L
import keras
import matplotlib.pyplot as plt

def get_model():

    '''    
    return:
        base model for training
    '''
    
    input = L.Input(shape = (48, 48, 1))
    x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = L.Conv2D(32, (3, 3), activation='relu')(x)
    x = L.Conv2D(64, (3, 3), activation='relu')(x)
    x = L.Dropout(0.5)(x)
    
    x = L.MaxPooling2D(pool_size=(3, 3))(x)
    
    x = L.Flatten(name = 'bottleneck')(x)
    x = L.Dense(64, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    output = L.Dense(7, activation='softmax')(x)
  
    model = keras.Model(input = input, output = output)
    
    model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_training(history, filename):
    '''
        polt the train data image
    '''
    
    output_acc = history.history['acc']
    val_output_acc = history.history['val_acc']

    output_loss = history.history['loss']
    val_output_loss = history.history['val_loss']
    
    epochs = range(len(val_output_acc))
    
    plt.figure()
    plt.plot(epochs, output_acc, 'b-', label='train accuracy')
    plt.plot(epochs, val_output_acc, 'r-', label='validation accuracy')
    plt.legend(loc='best')
    plt.title('Training and validation accuracy')
    plt.savefig(filename+'_accuray'+'.png')
    
    plt.figure()
    plt.plot(epochs, output_loss, 'b-', label='train loss')
    plt.plot(epochs,  val_output_loss, 'r-', label='validation loss')
    plt.legend(loc='best')
    plt.title('Training and validation loss')
    plt.savefig(filename+'_loss' + '.png')

if __name__ == '__main__':
    get_model()