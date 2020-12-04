# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 21:15:20 2020

@author: lzzzz
"""


import matplotlib.pyplot as plt
import numpy as np

#load the result of four different models

f1=np.load('result_ALL/proposed_re1.npy')
f2=np.load('result_ALL/CNN_biat.npy')
f3=np.load('result_ALL/CNN_LS.npy')
f4=np.load('result_ALL/chen1.npy')


abnor=['AF','I-AVB','LBBB','Normal','PAC','PVC','RBBB','STD','STE']
x=np.arange(len(abnor))
barWidth=0.18

b1=f1
b2=f2
b3=f3
b4=f4


r1 = np.arange(len(b1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4=  [x + barWidth for x in r3]

plt.bar(r1, b1, color='mediumblue', width=barWidth, edgecolor='white', label='Proposed_model')
plt.bar(r2, b2, color='orange', width=barWidth, edgecolor='white', label='CNN_attention_BIlSTM')
plt.bar(r3, b3, color='green', width=barWidth, edgecolor='white', label='CNN_LSTM')
plt.bar(r4, b4, color='red', width=barWidth, edgecolor='white', label='Challenge_best')

plt.xlabel('Label', fontweight='bold')
plt.ylabel('F1 score', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(b1))], ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC','PVC','RBBB','STD','STE'])
plt.title('')
# Create legend & Show graphic
plt.legend(loc='best',bbox_to_anchor=(1,1))
plt.show()




