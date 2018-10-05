# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:41:10 2018

@author: Yongrui Huang
"""

'''
    This is an example of handle EEG data in DEAP dataset, whose EEG features have been preprocessed.
    the shape of 'EEG.npy' stored an array with shape (60, 85)
    60 represents 60 seconds, the origin dataset is 128Hz and we get the average for each feature to avoid noise.
    85 represents 85 feature for each second. The meaning of each feature was presented in paper.
    The DEAP dataset contains 40 trials for each subject.
    In this example, for each subject, 20 trials are for training whereas the other trials are for testing. 
    This is an example of using 20 trials for training and the other for testing.
'''

import numpy as np
import sys
sys.path.append('../process_EEG')
import EEG_tool

if __name__ == '__main__':
    ROOT_PATH = '../../dataset/DEAP/'
    
    for subject_id in range(1, 23):
        subject_path = ROOT_PATH + str(subject_id)+'/'
        EEG_model = EEG_tool.EEG_model()
        
        #random select 20 trial for training, the other trials for testing
        train_idxs_set = set(np.random.choice(np.arange(1, 41), size = 20, replace = False))
        all_set = set(np.arange(1, 41))
        test_idxs_set = all_set - train_idxs_set
        
        #training
        for trial_id in train_idxs_set:
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            EEG_model.add_one_trial_data(trial_path, preprocessed = True)
            
        EEG_model.train()
        
        #testing
        acc_valence, acc_arousal = 0., 0.
        for trial_id in test_idxs_set:
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            valence_correct, arousal_correct = EEG_model.predict_one_trial(trial_path, preprocessed=True)
            acc_valence += 1 if valence_correct else 0
            acc_arousal += 1 if arousal_correct else 0
        
        print (subject_id, acc_valence/20, acc_arousal/20)
    
    
    
    