# Auto-detection-ECGs
Automatic detection for multi-labelled cardiac arrhythmia based on frame blocking pre-processing and Residual Networks

Requirements
The code is implemented based on python 3.6 with tensorflow==2.0.0 and keras 2.3.1. Please check Requirements.txt.

Scripts:
Preprocess.py: scripts for loading data and preprocessing. 
Proposed_model.py: scripts for traing, validation and testing on the proposed model.
CNN_LSTM.py: scripts for traing, validation and testing on the plain CNN+LSTM model.
CNN_attenBILSTM.py: scripts for traing, validation and testing on the plain CNN with attention-based BiLSTM model.
Challengebest_model.py: scripts for traing, validation and testing on the Challenge-best model on CPSC 2018.
Bar.py: scripts for plotting of F1 scores.
ROC.py: scripts for plotting of ROC curves.
