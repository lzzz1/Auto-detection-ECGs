# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:27:14 2020

@author: win7
"""

#Pre-processing and frame blocking
import scipy.io as sio
import os
import wfdb
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import random

#The path for orginal raw dataset (.mat file)
path='/Data/'
FileNames=os.listdir(path)
Record_set=[]


#split the ECG records based on the different labels
for i in FileNames:
    if i.endswith('.hea'):
        headname=os.path.splitext(i)
        recordName=headname[0]
        record=wfdb.rdheader(path+recordName)
        headdata=record.__dict__
        label=headdata['comments'][2][3:]
        label=label.split(',')
        if len(label)==1:
            if ' AF' in label: #detect the first label of that record
                Record_set.append(recordName)
        elif ' AF' in label or 'AF' in label:#detect the second or thrid label
             Record_set.append(recordName) 

#the function of noise-removing
Fstop1=35
fs=500    
def LowPassFilter(Input_sig,Fstop1): #butterworth loss-pass filter
    b, a = signal.butter(8, 2.0*Fstop1/fs, 'lowpass') 
    filtedData = signal.filtfilt(b, a, Input_sig)
    filtedData=np.float16(filtedData)
    return filtedData

#Frame blocking
Fl=2000    #frame length
Fn=10      #number of frames
def frame_split(signal,Fl,Fn):
    signal_length=len(signal[0])
    Fs=int(np.ceil((1.0*signal_length-Fl)/(Fn-1)))
    pad_length = int((Fn - 1)*Fs  + Fl)
    pad_signal=np.pad(signal,((0,0),(0,pad_length-signal_length)),'constant')
    pad_signal=pad_signal.T#.T will not break the sequence of ECG 
    indices = np.tile(np.arange(0, Fl), (Fn, 1)) + np.tile(np.arange(0, Fn*Fs, Fs), (Fl, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]  
    return frames


#Pre-processing of the raw data 
Sig_set_filted=[]           
for i in Record_set:
    inputsignal=sio.loadmat(path+i+'.mat')['val']
    filtered=LowPassFilter(inputsignal,Fstop1)
    frame_res=frame_split(filtered,Fl,Fn)
    Sig_set_filted.append(frame_res)

Sig_set_filted=np.asarray(Sig_set_filted)
np.save(r'\Input_set\Dataset_AF.npy',Sig_set_filted)
