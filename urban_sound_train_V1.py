#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:35:26 2020

@author: redcape
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]
sound_data = np.load('/home/redcape/Desktop/sound classification-v.0/urban_sound_train.npz')
X_data = sound_data['X']
y_data = sound_data['y']
groups = sound_data['groups']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # ensemble will create random train and test bag as per the n0. iteration specified

for train_idx, val_idx in gss.split(X_data, y_data, groups=groups):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    groups_train = groups[train_idx]

    X_val = X_data[val_idx]
    y_val = y_data[val_idx]
    groups_test = groups[val_idx]
    
np.intersect1d(groups_train, groups_test) # will return intersection between two array

X_train = tf.keras.utils.normalize(X_train, axis=1)
y_train = tf.keras.utils.normalize(X_train, axis=1)

training_epochs = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#last layer should have number of classfication(10) in the dence layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #for probability distiribution use softmmax

model.compile(optimizer= 'adam', 
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=1)
