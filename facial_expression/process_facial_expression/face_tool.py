# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:16:19 2018

@author: Yongrui Huang
"""

import numpy as np
import pandas as pd
from keras.optimizers import SGD
import keras
import keras.layers as L

def format_raw_images_data(imgs):
    '''
        conduct normalize and shape the image data in order to feed it directly to keras model
        Parameter:
            imgs: shape(?, 48, 48), all pixels are range from 0 to 255
        Return:
            shape(?, 48, 48, 1), image data after normalizing
        
    '''
    X_mean = np.load('../../dataset/fer2013/X_mean.npy')
    imgs = np.array(imgs) - X_mean
    return imgs.reshape(imgs.shape[0], 48, 48, 1)

def setup_to_finetune(model, freeze_num = 0):
    '''
        fix the convolution layer and prepare for the training of dense layer
        Parameter:
            model: the baseline model
            freeze_num: the number of the layers need to be fixed
    '''
    for layer in model.layers[:freeze_num]:
        layer.trainable = False
    for layer in model.layers[freeze_num:]:
        layer.trainable = True
     
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', \
                  metrics=['accuracy'])

def get_model():
    '''
      load the baseline model and return the new model
      Return:
          new_model: new model which the convolution layers are fixed.
    '''
    
    old_model = keras.models.load_model('../../model/CNN_expression_baseline.h5')
    feature_layer = old_model.get_layer('bottleneck').output
    x = L.Dense(64, activation='relu')(feature_layer)
    valence_output = L.Dense(1, activation='sigmoid', name = 'valence')(x)
    arousal_output = L.Dense(1, activation='sigmoid', name = 'arousal')(x)
    new_model = keras.Model(input = old_model.inputs, outputs = [valence_output, arousal_output])
    setup_to_finetune(new_model, freeze_num=3)
    return new_model

class Face_model:
    '''
        Attributes:
            model: model for classifying valence and arousal at the same time, built by keras
            X_faces: the list that saves all EEGs features
            y_valence: the valence label list, ground true
            y_arousal: the arousal label list, ground true
    '''
    model = None
    X = None
    y_valence = None
    y_arousal = None
    
    def __init__(self):
        self.model = get_model()
        self.X = []
        self.y_valence = []
        self.y_arousal = []
        
    def train(self):
        '''
            train the model
        '''
        
        #format data
        self.X = format_raw_images_data(self.X)
        #train model
        epochs = 12
        batch_size = 16
        self.model.fit(self.X, [self.y_valence, self.y_arousal], epochs=epochs, batch_size=batch_size, verbose=0)
    
        
    def add_one_trial_data(self, trial_path):
        '''
            read one-trial data from trial_path and put them into X, valence_y, arousal_y
            Parameter:
                trial_path: the file path of the trial
        '''
        
        #load trial data
        imgs = np.load(trial_path + 'faces.npy')
        label = pd.read_csv(trial_path + 'label.csv')
        
        #random pick 50 image for each trial
        idxs = np.random.choice(len(imgs), size = 50 if len(imgs) > 50 else len(imgs), replace = False)
        for idx in idxs:
            
            self.X.append(imgs[idx])
            self.y_valence.append(int(label['valence'] > 5))
            self.y_arousal.append(int(label['arousal'] > 5))
    
    def predict_one_trial(self, trial_path):
        '''
             use model to predict one trial
             Parameter:
                 trial_path: the trial's path
                 model: the trained model
             Return:
                 A: whether the valence was correctly predict. (1 stands for correct while 0 otherwise)
                 B: whether the arousal was correctly predict. (1 stands for correct while 0 otherwise)
         '''
         
        imgs = np.load(trial_path+'faces.npy')
        X = format_raw_images_data(imgs)
        label = pd.read_csv(trial_path + 'label.csv')
        [predict_valences, predict_arousals] = self.model.predict(X)
        predict_valence = np.sum(predict_valences)/float(len(predict_valences)) > 0.5
        predict_arousal = np.sum(predict_arousals)/float(len(predict_arousals)) > 0.5
        ground_true_valence = int(label['valence']) > 5
        ground_true_arousal = int(label['arousal']) > 5
         #print (predict_valence, ground_true_valence)
         #print (predict_arousal, ground_true_arousal)
         
        return (predict_valence == ground_true_valence), (predict_arousal == ground_true_arousal)

    def predict_one_trial_scores(self, trial_path):
        '''
             use model to predict one trial's scores
             Parameter:
                 trial_path: the trial's path
                 model: the trained model
             Return:
                 score_valence: the scores of valence predicted by face model
                 score_arousal: the scores of arousal predicted by EEG model
         '''
         
        imgs = np.load(trial_path+'faces.npy')
        X = format_raw_images_data(imgs)
        [predict_valences, predict_arousals] = self.model.predict(X)
        score_valence = np.sum(predict_valences)/float(len(predict_valences))
        score_arousal = np.sum(predict_arousals)/float(len(predict_arousals))
        
        return score_valence, score_arousal
         
    def predict_one_trial_results(self, trial_path):
        '''
             use model to predict one trial's scores
             Parameter:
                 trial_path: the trial's path
                 model: the trained model
             Return:
                 result_valence: the results of valence predicted by face model
                 result_arousal: the results of arousal predicted by EEG model
        '''   
        
        score_valence, score_arousal = self.predict_one_trial_scores(trial_path)
        result_valence = score_valence > 0.5
        result_arousal = score_arousal > 0.5
        
        return result_valence, result_arousal