#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:06:51 2020

@author: redcape
"""

import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
import numpy as np

#Load audio file
''' can also be done as
data, rate = wavfile.read('filename')'''
data, rate = librosa.load('/home/redcape/Desktop/UrbanSound/data/gun_shot/7060.wav')

#Noice reduction
noise_part = data[0:25000]
noise_reduce = nr.reduce_noise(audio_clip=data, noise_clip=noise_part, verbose=False)

#Visualize
print('raw file')
plt.plot(data)
print('noise reduced file')
plt.plot(noise_reduce)

#triming the scilence part from the audio
trimmed, index = librosa.effects.trim(noise_reduce, top_db=20, frame_length=512, hop_length=64)

#visualize
print("trimmed file")
plt.plot(trimmed)

#extrating absolute value short-time fourier transform
''' Since sound events have different durations(number of samples), the 2-d 
feature arrays are flattened using mean on the frequency axis. Thus, the audio
 clips will be represented using an array of fixed size 257 (number of STFT frequency bins). '''
 
stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))



