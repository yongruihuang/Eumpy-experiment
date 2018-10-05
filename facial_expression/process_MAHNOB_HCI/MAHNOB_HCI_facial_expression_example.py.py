# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:47:53 2018

@author: Yongrui Huang
"""

"""
  This is an example for processing facial expression data in MAHNOB_HCI dataset using the function in ../process_facial_expression/face_tool script
  Note that the dataset's format should be same as the format performed in 'dataset' folder
  The MAHNOB_HCI dataset contains 20 trials for each subject.
  In this example, a leave-one-out cross validation is performed for each subject.
"""

import sys
sys.path.append('../process_facial_expression')
import face_tool

if __name__ == '__main__':
    
    root_path = '../../dataset/' + 'MAHNOB_HCI/'  
    
    #for each subject we train 20 model
    for subject_id in range(1, 10):
        #calculate accuracy
        acc_valence, acc_arousal = 0, 0
        
        subject_path = root_path + str(subject_id) + '/'
        #each trial has one change to become validation set. leave-one-trial out
        for validation_trial_id in range(1, 21):
        
            face_model = face_tool.Face_model()
            #use other 19 trial as train set
            for train_trial_id in range(1, 21):
                #can't put the validation trial into train set
                if train_trial_id == validation_trial_id:
                    continue

                #load train data
                path = subject_path + 'trial_' + str(train_trial_id) + '/'
                face_model.add_one_trial_data(path)
            
            face_model.train()
            
            #validation on one trial
            path = subject_path + 'trial_' + str(validation_trial_id) + '/'
            
            #predict one trial
            valence_correct, arousal_correct = face_model.predict_one_trial(path)
            print (valence_correct, arousal_correct)
            if(valence_correct):
                acc_valence+=1
            if(arousal_correct):
                acc_arousal+=1
                
        print ('subject: ' + str(subject_id))
        print (acc_valence/20., acc_arousal/20.)
            
            
            
            
            
            
            
            
            