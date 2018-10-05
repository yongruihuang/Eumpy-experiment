# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:37:54 2018

@author: Yongrui Huang
"""

"""
  This is an example for processing enum fusion algorithm in ONLINE dataset using the function in ../process_enum/enum_tool script
  Note that the dataset's format should be same as the format performed in 'dataset' folder
  The ONLINE dataset contains 40 trials for each subject.
  In this example, for each subject, 20 trials are for training whereas the other trials are for testing. 
"""

import sys
sys.path.append('../process_enum')
import numpy as np
import enum_tool

if __name__ == '__main__':
    ROOT_PATH = '../../dataset/ONLINE/'
    
    for subject_id in range(1, 23):
        subject_path = ROOT_PATH + str(subject_id)+'/'
        
        enum_model = enum_tool.Enum_model(preprocessed=True)
        #random select 20 trial for training, the other trials for testing
        train_idxs_set = set(np.random.choice(np.arange(1, 41), size = 20, replace = False))
        all_set = set(np.arange(1, 41))
        test_idxs_set = all_set - train_idxs_set
        
        #training
        for trial_id in train_idxs_set:
            
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            enum_model.add_one_trial_data(trial_path)
        
        enum_model.train()
        
        #testing
        acc_valence, acc_arousal = 0., 0.
        for trial_id in test_idxs_set:
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            valence_correct, arousal_correct = enum_model.predict_one_trial(trial_path)
            acc_valence += 1 if valence_correct else 0
            acc_arousal += 1 if arousal_correct else 0
        
        print (subject_id, acc_valence/20, acc_arousal/20)