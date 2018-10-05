# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:01:31 2018

@author: Yongrui Huang
"""

'''
      This script describe the implement of using adaboost algorithm to combine 
    facial expression and EEG.
      The key goal of training for this approach is to train face_weight and 
    eeg_weight in order to obtain results using (1)
      result = 1./(e^(-(face_weight*face_score+eeg_weight*eeg_score)) (1)
    where result is high when result > 0.5 else low
      
      In that case, to get face_weight and eeg_weight, we follow these steps:
          Step 1:
              Initialize the data weights for each data point by (2)
                      data_weight = 1/M   (2)
              where M is the number of the data sample
         Step 2:
             Use subclassifier to predict the training set and calculate the 
             error rate by (3)
                     error_rate = sum(data_weight * (y_hat!=y)) (3)
             where y_hat is the label predicted by sub-classifier
             (i.e. facial expression or eeg) and y is the ground truth label
         Step 3:
             Update the weight of the sub-classifier by (4)
                 sub_weight = 1./2*((1-error_rate)/error_rate) (4)
             where sub_weight is face_weight or eeg_weight
         Step 4:
             Update data weight by (5)
                 data_weight = data_weight * exp(-sub_weight)/sum(data_weight)
                 if y_hat is y else 
                 data_weight * exp(sub_weight)/sum(data_weight)   (5)
                 
         Continue to calculate the weight of the subsequent subclassifier, i.e
         return to Step 2
              
'''

import sys
sys.path.append('../../EEG/process_EEG')
sys.path.append('../../facial_expression/process_facial_expression')
import face_tool
import EEG_tool
import numpy as np
import pandas as pd

def read_label(trial_path):
    '''
        read trial's ground truth from trial path
        Parameter:
            trial_path: the path of the trial file
        Returens:
            ground_true_valence: the ground truth value for this trial in valence space 
            ground_true_arousal: the ground truth value for this trial in arousal space
    '''
    
    label = pd.read_csv(trial_path + 'label.csv')
    ground_true_valence = int(label['valence']) > 5
    ground_true_arousal = int(label['arousal']) > 5
    
    return ground_true_valence, ground_true_arousal

def init_data_weight(num_sample):
    '''
        Step 1: initialize the data weights
        Patameters:
            num_sample: the number of the sample
        Returns:
            valence_data_weight: the data weight for sample in valence space
            arousal_data_weight: the data weight for sample in arousal space
    '''
    valence_data_weight = np.zeros((num_sample,)) + 1./num_sample
    arousal_data_weight = np.zeros((num_sample,)) + 1./num_sample
    return valence_data_weight, arousal_data_weight

def cal_error_rate(data_weight, y_hat, y):
    '''
        Step 2: calculate the error rate
        Parameters:
            data_weight: the weights of data, shape (1, ?), array-like numpy
            y_hat: the label predicted by sub-classifier, shape (1, ?), array-like numpy
            y: the grounp truth label, shape (1, ?), array-like numpy
        Returns:
            error_rate: a scalar shows the number of the samples that were 
            mis-classified by classifier in the dataset
        In the real program, add a very small number to prevent it divided by zero
    '''
    bias = 1e-11
    return sum(data_weight * (y_hat!=y)) + bias

def compute_sub_classifier_weight(error_rate):
    '''
        Step 3:
        Parameters:
            error_rate: a scalar shows the number of the samples that were 
            mis-classified by classifier in the dataset
        Returns:
            The weights of the sub-classifier, scalar 
    '''
    return 1./2*((1-error_rate)/error_rate)

def update_data_weight(data_weights, y_hat, y, classifier_weight):
    '''
        Step 4:
            Parameters:
                data_weight: the weights of data, shape (1, ?), array-like numpy
                y_hat: the label predicted by sub-classifier, shape (1, ?), array-like numpy
                y: the grounp truth label, shape (1, ?), array-like numpy
                classifier_weight: the weights of the sub-classifier
            Returns:
                The new data_weight, scalar
            
    '''
    symbol = (y != y_hat)
    idx_0 = np.where(symbol == 0)[0]
    symbol[idx_0] = -1
    
    data_weights *= np.exp(symbol*classifier_weight)
    return data_weights

class Adaboost_model:
    
    '''
        Attributes:
            face_model: model for predicting face data
            EEG_model: model for predicting EEG data
            face_valence_weight: the linear wight of valence for face model
            face_arousal_weight: the linear wight of arousal for face model
            eeg_valence_weight: the linear wight of valence for eeg model
            eeg_arousal_weight: the linear wight of arousal for eeg model
            train_trial_paths: a list stored all train trial's path
            
            The equation (1) shown as fllowed was used to calculated and updated the adaboost model
                1./(e^(-(face_weight*face_score+eeg_weight*eeg_score))        (1)
            where face_weight is face_valence_weight when the model calculate the valence space
            and etc...
    '''
    
    face_model = None
    EEG_model = None
    face_valence_weight = 0.5
    face_arousal_weight = 0.5
    eeg_valence_weight = 0.5
    eeg_arousal_weight = 0.5
    train_trial_paths = None
    preprocessed = None
    
    def __init__(self, preprocessed):
        self.preprocessed = preprocessed
        self.face_model = face_tool.Face_model()
        self.EEG_model = EEG_tool.EEG_model()
        self.train_trial_paths = []
        
    def train(self):
        '''
            train face_model and EEG_model
            In the same time, calculate the weights
        '''
        #train face_model and EEG_model
        self.face_model.train()
        self.EEG_model.train()
        
        #add all train trial results into list
        face_valence_results, face_arousal_results, eeg_valence_results, eeg_arousal_results = [], [], [], []
        valences, arousals = [], []
        for trial_path in self.train_trial_paths:
            #process face
            face_valence_result, face_arousal_result = self.face_model.predict_one_trial_results(trial_path)
            face_valence_results.append(face_valence_result)
            face_arousal_results.append(face_arousal_result)
            
            #process EEG
            eeg_valence_result, eeg_arousal_result = self.EEG_model.predict_one_trial_results(trial_path, self.preprocessed)
            eeg_valence_results.append(eeg_valence_result)
            eeg_arousal_results.append(eeg_arousal_result)
             
            #process label
            valence, arousal = read_label(trial_path)
            valences.append(valence)
            arousals.append(arousal)
    
        #trun them into numpy array
        face_valence_results = np.array(face_valence_results)
        face_arousal_results = np.array(face_arousal_results)
        eeg_valence_results = np.array(eeg_valence_results)
        eeg_arousal_results = np.array(eeg_arousal_results)
        valences = np.array(valences)
        arousals = np.array(arousals)
        
        #Step 1: initialize data weight
        M = len(face_valence_results)
        valence_data_weight, arousal_data_weight = init_data_weight(M)
        
        #Step 2: compute the error rate
        valence_error_rate = cal_error_rate(valence_data_weight, face_valence_results, valences)
        arousal_error_rate = cal_error_rate(arousal_data_weight, face_arousal_results, arousals)
        
        ##Step 3: compute face weight
        self.face_valence_weight = compute_sub_classifier_weight(valence_error_rate)
        self.face_arousal_weight = compute_sub_classifier_weight(arousal_error_rate)
        
        #Step 4: update data weight
        valence_data_weight = update_data_weight(valence_data_weight, face_valence_results, valences, self.face_valence_weight)
        arousal_data_weight = update_data_weight(arousal_data_weight, face_arousal_results, arousals, self.face_arousal_weight)
        
        #Reuten to Step 2: compute the error rate for valence 
        valence_error_rate = cal_error_rate(valence_data_weight, eeg_valence_results, valences)
        arousal_error_rate = cal_error_rate(arousal_data_weight, eeg_arousal_results, arousals)
        
        #Step 3: compute eeg weight
        self.eeg_valence_weight = compute_sub_classifier_weight(valence_error_rate)
        self.eeg_arousal_weight = compute_sub_classifier_weight(arousal_error_rate)
        
        
    def add_one_trial_data(self, trial_path):
        '''
            read one-trial data from trial_path and put them into face_model, EEG_model
            Parameter:
                trial_path: the file path of the trial
                preprocessed: whether the EEG data is preprocessed
        '''
        self.face_model.add_one_trial_data(trial_path)
        self.EEG_model.add_one_trial_data(trial_path, preprocessed = self.preprocessed)
        self.train_trial_paths.append(trial_path)
    
    def predict_one_trial(self, trial_path):
        '''
             use model to predict one trial
             Parameter:
                 trial_path: the trial's path
                 preprocessed: whether the EEG data is preprocessed
             Return:
                 A: whether the valence was correctly predict. (1 stands for correct 0 otherwise)
                 B: whether the arousal was correctly predict. (1 stands for correct 0 otherwise)
        '''
        
        #load face data
        face_valence_result, face_arousal_result = self.face_model.predict_one_trial_results(trial_path)
         
        #load EEG data
        eeg_valence_result, eeg_arousal_result = self.EEG_model.predict_one_trial_results(trial_path, preprocessed = self.preprocessed) 
        
        #predict result in valence space
        face_valence_result = (1 if face_valence_result == 1 else -1)
        eeg_valence_result = (1 if eeg_valence_result == 1 else -1)
        predict_valence = (self.face_valence_weight * face_valence_result + self.eeg_valence_weight * eeg_valence_result) > 0
        
        #predict result in arousal space
        face_arousal_result = (1 if face_arousal_result == 1 else -1)
        eeg_arousal_result = (1 if eeg_arousal_result == 1 else -1)
        predict_arousal = (self.face_arousal_weight * face_arousal_result + self.eeg_arousal_weight * eeg_arousal_result) > 0
    
        #load ground truth result for valence and arousal
        valence, arousal = read_label(trial_path)
        
        return valence == predict_valence, arousal == predict_arousal
        