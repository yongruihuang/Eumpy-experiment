# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:55:02 2018

@author: Yongrui Huang
"""

from sklearn.svm.classes import SVC
import numpy as np
import pandas as pd
import mne

def extract_EEG_feature(raw_EEG_obj):
    '''
        Extract PSD feature from raw EEG data
        Parameter:
            raw_EEG_obj: raw objects from mne library
        Rerturn:
            average_feature: extracted feature
    
    '''
    
    #select 14 electrodes
    EEG_raw = raw_EEG_obj.pick_channels(['Fp1', 'T7', 'CP1', 'Oz', 'Fp2', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'PO4'])
   
    EEG_data_frame = EEG_raw.to_data_frame()
    
    #calculate three symmetric pairs
    EEG_data_frame['T7-T8'] = EEG_data_frame['T7'] - EEG_data_frame['T8']
    EEG_data_frame['Fp1-Fp2'] = EEG_data_frame['Fp1'] - EEG_data_frame['Fp2']
    EEG_data_frame['CP1-CP2'] = EEG_data_frame['CP1'] - EEG_data_frame['CP2']
    
    #extract PSD feature from different frequecy
    EEG_raw_numpy = np.array(EEG_data_frame).T
    EEG_theta = mne.filter.filter_data(EEG_raw_numpy, sfreq = 256, l_freq=4, h_freq=8, verbose = 'ERROR')
    EEG_slow_alpha = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=8, h_freq=10, verbose = 'ERROR')
    EEG_alpha = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=8, h_freq=12, verbose = 'ERROR')
    EEG_beta = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=12, h_freq=30, verbose = 'ERROR')
    EEG_gamma = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=30, h_freq=4, verbose = 'ERROR')
    
    #concat them together
    features = np.concatenate((EEG_theta, EEG_slow_alpha, EEG_alpha, EEG_beta, EEG_gamma), axis=0)
    
    #get average in each second for decreasing noise and reduce the number of sample for quicker training.
    left_idx = 0
    len_features = features.shape[1]
    features_list = []
    while left_idx < len_features:
        sub_features = features[:, left_idx:left_idx+256] if left_idx+256 < len_features else features[:, left_idx:]
        features_list.append(np.average(sub_features, axis = 1))
        left_idx += 256
    average_feature = np.array(features_list)
    
    return average_feature

class EEG_model:
    '''
        This class allow EEG model become an independent model like facial 
        expression model rathan than two separated model.
        Attributes:
            valence_model: model for classifying valence
            arousal_model: model for classifying arousal
            X: the list that saves all EEGs features
            y_valence: the valence label list, ground true
            y_arousal: the arousal label list, ground true
    '''
    
    valence_model = None
    arousal_model = None
    X = None
    y_valence = None
    y_arousal = None
    
    def __init__(self):
        self.valence_model = SVC(C = 15)
        self.arousal_model = SVC(C = 15)
        self.X = []
        self.y_valence = []
        self.y_arousal = []
        
    def train(self):
        '''
            train valence_model and arousal_model using EEG data
        '''
        self.valence_model.fit(self.X, self.y_valence)
        self.arousal_model.fit(self.X, self.y_arousal)
        
    def add_one_trial_data(self, trial_path, preprocessed = False):
        '''
        read one-trial
        data from trial_path and put them into X, valence_y, arousal_y
        Parameter:
            trial_path: the file path of the trial
            preprocessed: whether the EEG data is preprocessed
            
        '''
        
        #load EEG data
        if preprocessed is False:
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
        else:
            EEGs = np.load(trial_path + 'EEG.npy')
        label = pd.read_csv(trial_path + 'label.csv')
        
        for EEG in EEGs:
            self.X.append(EEG)
            self.y_valence.append(int(label['valence'] > 5))
            self.y_arousal.append(int(label['arousal'] > 5))
            
            
    def predict_one_trial(self, trial_path, preprocessed = False):
         '''
             use model to predict one trial
             Parameter:
                 trial_path: the trial's path
                 preprocessed: whether the EEG data is preprocessed
             Return:
                 A: whether the valence was correctly predict. 
                 (1 stands for correct 0 otherwise)
                 B: whether the arousal was correctly predict. 
                 (1 stands for correct 0 otherwise)
        '''
        
        #load trial data
         if preprocessed is False:
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
         else:
            EEGs = np.load(trial_path + 'EEG.npy')
             
         label = pd.read_csv(trial_path + 'label.csv')
         predict_valences, predict_arousals = self.valence_model.predict(EEGs), self.arousal_model.predict(EEGs)
         predict_valence = np.sum(predict_valences)/float(len(predict_valences)) > 0.5
         predict_arousal = np.sum(predict_arousals)/float(len(predict_arousals)) > 0.5
         ground_true_valence = int(label['valence']) > 5
         ground_true_arousal = int(label['arousal']) > 5
         
         return (predict_valence == ground_true_valence), (predict_arousal == ground_true_arousal)
     
        
    def predict_one_trial_scores(self, trial_path, preprocessed = False):
        '''
             use model to predict one trial
             Parameter:
                 trial_path: the trial's path
                 preprocessed: whether the EEG data is preprocessed
             Return:
                 score_valence: the scores of valence predicted by face model
                 score_arousal: the scores of arousal predicted by EEG model
        '''
        #load trial data
        if preprocessed is False:
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
        else:
            EEGs = np.load(trial_path + 'EEG.npy')
         
        predict_valences, predict_arousals = self.valence_model.predict(EEGs), self.arousal_model.predict(EEGs)
         
        score_valence = np.sum(predict_valences)/float(len(predict_valences))
        score_arousal = np.sum(predict_arousals)/float(len(predict_arousals))
         
        return score_valence, score_arousal
    
    def predict_one_trial_results(self, trial_path, preprocessed = False):
        '''
             use model to predict one trial
             Parameter:
                 trial_path: the trial's path
                 preprocessed: whether the EEG data is preprocessed
             Return:
                 result_valence: the results of valence predicted by face model
                 result_arousal: the results of arousal predicted by EEG model
        '''
        score_valence, score_arousal = self.predict_one_trial_scores(trial_path, preprocessed)
        result_valence = score_valence > 0.5
        result_arousal = score_arousal > 0.5
        
        return result_valence, result_arousal