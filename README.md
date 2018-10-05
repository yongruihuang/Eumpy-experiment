# Eumpy-experiment
This is a project of combining facial expression and EEG for emotion recognition using python in emotion experiment.
Data from emotion experiment, such as [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/) are suitable for using this tool to processce.

## Prerequisites

What things you need to install the software and how to install them

- [Anaconda (Python 3.7 version)](https://www.anaconda.com/download/#windows)
- [Keras 2.2.4](https://pypi.org/project/Keras/)
- [MNE](https://www.martinos.org/mne/stable/install_mne_python.html)

## Note
Before processing the data, the structure of data should be organized in the following format and put it under the directory 'dataset'.

dataset
  |- MAHNOB-HCI
  | |- 1
  | | |- trial_1
  | | | |- faces.npy
  | | | |- EEG.npy
  | | | |- label.csv
  | | |- trial_2
  | | |- trial_...
  | |- 2
  | | |- trial_1
  | | |- trial_2
  | | |- trial_..


Where MAHNOB-HCI represents the name of the dataset, '1' represents the id of the subject and trial_1 represets one trial. In each trial, 3 files are presented. 'faces.npy' contains the facial expression data, in our dataset, it is a numpy array-like with shape (?, 48, 48). 'label.csv' contains the ground truth label for this trial. 

As for EEG data, when it presents itself as 'EEG.npy', it means that the data is preprocessed and the data can be directly feed to classifier. However, when it is presented as 'EEG.raw.fif', it means that the data is not preprocessed. We have provided some preprocess approach and the user can choose to use our approach or theirs.
In our released dataset, the dataset DEAP and ONLINE (recorded by us) are used precessed EEG data while the MAHNOB-HCI dataset is used raw EEG data.

Due to the limit of uploading file in Github, we put our dataset in [here](https://pan.baidu.com/s/1a6k5_tRXk3niZqrxXyNIhg), password 7dep.
Please download the dataset and put it under the directory 'dataset'.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



