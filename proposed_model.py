# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import keras
import random
from keras.layers import Input, Add, Dense, Activation,BatchNormalization, Flatten, Conv1D, AveragePooling1D, MaxPooling1D
from keras import Model
from keras.layers import LSTM, Bidirectional,TimeDistributed
from keras_multi_head import MultiHeadAttention
from keras.layers import Dropout
from keras import regularizers
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.initializers import he_normal

dataset1=np.load(r'/Input_set/Dataset_AF.npy')
dataset2=np.load(r'/Input_set/Dataset_NonAF.npy')


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
#X_train, X_test, y_train, y_test = train_test_split(dataset1, labels1, test_size=0.2, random_state=42)

#Dense Block 1
def identity_block(X,f, filters):
    f1,f2=filters
    X_shortcut = X
    X=TimeDistributed(Conv1D(f1,f,strides=1,padding="same",kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01)))(X)
    X=BatchNormalization()(X)
    X = Activation('relu')(X)
    #X = Dropout(0.5)(X)
    
        
    X=TimeDistributed(Conv1D(f2,f,strides=1,padding="same",kernel_initializer = he_normal(seed=None)))(X)
    X=BatchNormalization()(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
    
#Dense Block2
def convolution_block(X,f, filters):
    f1,f2,f3=filters
    X_shortcut = X
    X=TimeDistributed(Conv1D(f1,f,strides=2,padding="same",kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01)))(X)
    X=BatchNormalization()(X)
    X = Activation('relu')(X)
    #X = Dropout(0.5)(X)
    
    X=TimeDistributed(Conv1D(f2,f,strides=1,padding="same",kernel_initializer = he_normal(seed=None)))(X)
    X=BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X_shortcut = TimeDistributed(Conv1D(f3,7, strides = 2,padding="same",kernel_initializer = he_normal(seed=None)))(X_shortcut)
    X_shortcut=BatchNormalization()(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X


def ResNet_small(input_shape):
    X_input = Input(input_shape)
    X=TimeDistributed(Conv1D(32,3,strides=2,padding="same",kernel_initializer = he_normal(seed=None),kernel_regularizer=regularizers.l2(0.01)))(X_input)
    X=BatchNormalization()(X)
    X = Activation('relu')(X)
    X = TimeDistributed(MaxPooling1D(3,strides = 2,padding="same"))(X)
    X = Dropout(0.5)(X)
    
    X=convolution_block(X,3,[32,64,64])
    X=identity_block(X,1,[32,64])
    X = Dropout(0.5)(X)
    X=convolution_block(X,3,[64,128,128])
    X=identity_block(X,3,[64,128])
    X=identity_block(X,3,[64,128])
    X = TimeDistributed(AveragePooling1D(pool_size = 2,padding = 'same'))(X)
    X = TimeDistributed(Flatten())(X)
    X = Dropout(0.5)(X)
    X=Bidirectional(LSTM(128,kernel_regularizer=regularizers.l2(0.01),return_sequences=True))(X)
    X = Dropout(0.5)(X)
    X=MultiHeadAttention(head_num=256)(X)
    X=LSTM(256,activation='relu',kernel_regularizer=regularizers.l2(0.01))(X)
    X=Dense(1,activation='sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'ResNet_small')
    return model

model = ResNet_small(input_shape = (10, 2000, 12))
adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()
model.fit(X_train2, y_train2,
          batch_size=64, epochs=100,validation_data=(X_val, y_val))

y_pre=model.predict(X_test)


#save the result for ROC curve, AUC and F1 scores
np.save(r'/result/y_preaf.npy',y_pre)
np.save(r'/result/y_testaf.npy',y_test)
