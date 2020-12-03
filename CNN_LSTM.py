# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:57:19 2020

@author: win7
"""

#Plain CNN + LSTM
import numpy as np
import keras
import random
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,Activation,Embedding
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional,TimeDistributed
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
from keras.initializers import he_normal

#
dataset1=np.load(r'/content/drive/My Drive/challenge_2020_DNN/Dataset_AF.npy')
dataset2=np.load(r'/content/drive/My Drive/challenge_2020_DNN/Dataset_NonAF.npy')

dataset_inp=np.vstack((dataset1,dataset2))

labels=[]
for i in range(len(dataset1)-len(dataset2)):
    labels.append(1)
for k in range(len(dataset2)):
    labels.append(0)
labels=np.asarray(labels)   

X_train, X_test, y_train, y_test = train_test_split(dataset_inp, labels, test_size=0.2, random_state=42)

kf= StratifiedKFold(n_splits=5,shuffle=True)
for train_index, test_index in kf.split(X_train,y_train):
    
    X_train2, X_val = dataset_inp[train_index], dataset_inp[test_index]
    y_train2, y_val = labels[train_index], labels[test_index]

model = Sequential()
model.add(TimeDistributed(Conv1D(32,3,strides=2,kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01)),input_shape=(10,2000,12)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(32,3,strides=2,kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(32,3,strides=2,kernel_initializer = he_normal(seed=None))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(64,3,strides=2,kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(64,3,strides=2,kernel_initializer = he_normal(seed=None))))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(64,3,strides=2,kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(128,3,strides=2,kernel_initializer = he_normal(seed=None))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(TimeDistributed(Conv1D(128,3,strides=2,kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01))))
model.add(BatchNormalization())
model.add(TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(LSTM(128, kernel_regularizer=regularizers.l2(0.01),return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()
model.fit(X_train2, y_train2,
          batch_size=64, epochs=100,validation_data=(X_val, y_val))

y_pre=model.predict_proba(X_test)


np.save(r'/content/drive/My Drive/challenge_2020_DNN/y_preaf3.npy',y_pre)
np.save(r'/content/drive/My Drive/challenge_2020_DNN/y_testaf3.npy',y_test)
