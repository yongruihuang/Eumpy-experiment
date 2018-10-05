# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:51:12 2018

@author: Yongrui Huang
"""

import sys
sys.path.append('../../EEG/process_EEG')
sys.path.append('../../facial_expression/process_facial_expression')
import face_tool
import EEG_tool
import numpy as np
import pandas as pd

def read_label(trial_path):
    '''
        read trial's ground true from trial path
        Parameter:
            trial_path: the path of the trial file
    '''
    label = pd.read_csv(trial_path + 'label.csv')
    ground_true_valence = int(label['valence']) > 5
    ground_true_arousal = int(label['arousal']) > 5
    
    return ground_true_valence, ground_true_arousal

def cal_mistake(predict_valences, predict_arousals, valences, arousals):
    '''
        calculate the number of the mistake in target weight
        Parameter:
            predict_valences: the array stroed predict valence, numpy-array-like
            predict_arousals: the array stroed predict arousal, numpy-array-like
            valences: the ground truth valence, numpy-array-like
            arousals: the ground truth arousal, numpy-array-like
        Return:
            valence_mistake: the number of mistake model taken in valence space
            arousal_mistake: the number of mistake model taken in arousal space
    '''
    valence_mistake, arousal_mistake = 0, 0
    for (predict_valence, predict_arousal, valence, arousal) in zip(predict_valences, predict_arousals, valences, arousals):
        if (predict_valence != valence):
            valence_mistake += 1
        if (predict_arousal != arousal):
            arousal_mistake += 1
    
    return valence_mistake, arousal_mistake

class Enum_model():
    '''
        Attributes:
            face_model: model for predicting face data
            EEG_model: model for predicting EEG data
            valence_weight: the linear wight of valence for two model
            arousal_weight: the linear wight of arousal for two model
            train_trial_paths: a list stored all train trial's path
    '''
    
    face_model = None
    EEG_model = None
    valence_weight = 0.5
    arousal_weight = 0.5
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
            In the same time, get the best weight
        '''
        
        # train face_model and EEG_model
        self.face_model.train()
        self.EEG_model.train()
        
        # find the best weight
        
        #calculate the scores and read label into list for each trial
        face_valence_scores, face_arousal_scores, eeg_valence_scores, eeg_arousal_scores = [], [], [], []
        valences, arousals = [], []
        for train_trial_path in self.train_trial_paths:
            
            face_valence_score, face_arousal_score = self.face_model.predict_one_trial_scores(train_trial_path)
            eeg_valence_score, eeg_arousal_score = self.EEG_model.predict_one_trial_scores(train_trial_path, self.preprocessed)
            valence, arousal = read_label(train_trial_path)
            
            face_valence_scores.append(face_valence_score)
            face_arousal_scores.append(face_arousal_score)
            eeg_valence_scores.append(eeg_valence_score)
            eeg_arousal_scores.append(eeg_arousal_score)
            valences.append(valence)
            arousals.append(arousal)
        
        face_valence_scores = np.array(face_valence_scores)
        face_arousal_scores = np.array(face_arousal_scores)
        eeg_valence_scores = np.array(eeg_valence_scores)
        eeg_arousal_scores = np.array(eeg_arousal_scores)
        valences = np.array(valences)
        arousals = np.array(arousals)
    
        #find the wight
        weights = np.arange(0, 1 + 1./100., 1./100.) 
        min_valence_mistake, min_arousal_mistake = 20, 20
        for weight in weights:
            predict_valences_scores = weight*face_valence_scores + (1-weight)*eeg_valence_scores
            predict_arousals_scores = weight*face_arousal_scores + (1-weight)*eeg_arousal_scores
            
            predict_valences = predict_valences_scores > 0.5
            predict_arousals = predict_arousals_scores > 0.5
            valence_mistake, arousal_mistake = cal_mistake(predict_valences, predict_arousals, valences, arousals)
            
            #print valence_mistake,arousal_mistake
            if valence_mistake <= min_valence_mistake:
                min_valence_mistake = valence_mistake
                self.valence_weight = weight
            if arousal_mistake <= min_arousal_mistake:
                min_arousal_mistake = arousal_mistake
                self.arousal_weight = weight
    
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
        face_valence_score, face_arousal_score = self.face_model.predict_one_trial_scores(trial_path)
         
        #load EEG data
        eeg_valence_score, eeg_arousal_score = self.EEG_model.predict_one_trial_scores(trial_path, preprocessed = self.preprocessed) 
        
        #calculate output
        valence_score = self.valence_weight*face_valence_score + (1-self.valence_weight)*eeg_valence_score
        arousal_score = self.arousal_weight*face_arousal_score + (1-self.arousal_weight)*eeg_arousal_score
            
        ground_truth_valence, ground_truth_arousal = read_label(trial_path)
         
        return (valence_score > 0.5) == ground_truth_valence, (arousal_score > 0.5) == ground_truth_arousal
        
    
    