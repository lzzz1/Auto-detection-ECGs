# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:33:11 2020

@author: lzzzz
"""
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

#load the result of abnormality
ypre1=np.load('y_preaf.npy')
ytest1=np.load('y_testaf.npy')
ypre2=np.load('plainCNN_attenBIL/y_preaf2.npy')
ytest2=np.load('plainCNN_attenBIL/y_testaf2.npy')
ypre3=np.load('CNN_LSTM/y_preaf3.npy')
ytest3=np.load('CNN_LSTM/y_testaf3.npy')
ypre4=np.load('chen/y_preaf_chen1.npy')
ytest4=np.load('chen/y_testaf_chen1.npy')

fpr_rt_lm1, tpr_rt_lm1, _ = roc_curve(ytest1, ypre1)
roc_auc1=auc(fpr_rt_lm1, tpr_rt_lm1)
fpr_rt_lm2, tpr_rt_lm2, _ = roc_curve(ytest2, ypre2)
roc_auc2=auc(fpr_rt_lm2, tpr_rt_lm2)
fpr_rt_lm3, tpr_rt_lm3, _ = roc_curve(ytest3, ypre3)
roc_auc3=auc(fpr_rt_lm3, tpr_rt_lm3)
fpr_rt_lm4, tpr_rt_lm4, _ = roc_curve(ytest4, ypre4)
roc_auc4=auc(fpr_rt_lm4, tpr_rt_lm4)
plt.figure()
plt.plot(fpr_rt_lm1, tpr_rt_lm1,label='Proposed_model on CPSC 2020(area = {0:0.2f})' ''.format(roc_auc1),color='blue')
plt.plot(fpr_rt_lm2, tpr_rt_lm2,label='challenge_best on CPSC 2020(area = {0:0.2f})' ''.format(roc_auc2),color='orange')
plt.plot(fpr_rt_lm3, tpr_rt_lm3,label='CNN_LSTM(area = {0:0.2f})' ''.format(roc_auc3),color='green')
plt.plot(fpr_rt_lm4, tpr_rt_lm4,label='Challenge_best(area = {0:0.2f})' ''.format(roc_auc4),color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--')

plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC of PVC')
plt.legend(loc='lower right')
plt.show()
