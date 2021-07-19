# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import json
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from utils.utils_preprocessing import preprocess_data
#%% LOAD DATA
folder = 'data/data_raw/'
Raw_data = {
    'Temperature sensor 1' : np.loadtxt(folder + "TS1.txt"),
    'Temperature sensor 2' : np.loadtxt(folder + "TS2.txt"),
    'Temperature sensor 3' : np.loadtxt(folder + "TS3.txt"),
    'Temperature sensor 4' : np.loadtxt(folder + "TS4.txt"),
    'Vibration sensor' : np.loadtxt(folder + "VS1.txt"),
    'Cooling efficiency' : np.loadtxt(folder + "CE.txt"),
    'Cooling power' : np.loadtxt(folder + "CP.txt"),
    'Efficiency factor' : np.loadtxt(folder + "SE.txt"),
    'Flow sensor 1' : np.loadtxt(folder + "FS1.txt"),
    'Flow sensor 2' : np.loadtxt(folder + "FS2.txt"), #10
    'Pressure sensor 1' : np.loadtxt(folder + "PS1.txt"),
    'Pressure sensor 2' : np.loadtxt(folder + "PS2.txt"),
    'Pressure sensor 3' : np.loadtxt(folder + "PS3.txt"),
    'Pressure sensor 4' : np.loadtxt(folder + "PS4.txt"),
    'Pressure sensor 5' : np.loadtxt(folder + "PS5.txt"),
    'Pressure sensor 6' : np.loadtxt(folder + "PS6.txt"),
    'Motor power' : np.loadtxt(folder + "EPS1.txt")
}

#labels load
labels_folder = 'data/labels/'
conditions = {
    'cooler' : np.loadtxt(labels_folder + 'labels_cooler.txt'),
    'valve' : np.loadtxt(labels_folder + 'labels_valve.txt'),
    'pump' : np.loadtxt(labels_folder + 'labels_pump.txt'),
    'accumulator' : np.loadtxt(labels_folder + 'labels_accumulator.txt'),
    'stableFlag' : np.loadtxt(labels_folder + 'labels_stableFlag.txt')
    }
#%%
condition = 'accumulator'
win_olap_str = 'win60_olap0'
t_win = int(win_olap_str.split('_')[0][3:])
t_olap = int(win_olap_str.split('_')[0][4:])
time_param = 'RMS'


sets = preprocess_data(Raw_data, conditions[condition], time_param,
                       t_win, t_olap)
X_train, X_test = sets[0].values, sets[1].values
Y_train, Y_test = sets[2], sets[3]
#%%
layers_dim = (20, 20, 20, 20, 20)
act_func = 'relu'
Model = MLPClassifier(hidden_layer_sizes = layers_dim, activation = act_func,
                      learning_rate = 'constant', early_stopping = True, 
                      n_iter_no_change = 200, max_iter = 1000)

Model.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, Model.predict(X_test))
#%%

