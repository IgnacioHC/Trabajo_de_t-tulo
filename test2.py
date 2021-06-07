# -*- coding: utf-8 -*-
"""
@author: Ignacio
"""
#%% Imports
import pandas as pd
import numpy as np

from test_forward_selection import get_combinations
from test_forward_selection import get_max_idxs
from test_forward_selection import get_XY_TrainTest

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#%% Load data
folder = 'data/data_TimeParams/'
TimeParams = {
    'RMS' : pd.read_csv(folder + 'data_RMS.csv'),
    'Peak2Peak' : pd.read_csv(folder + 'data_P2P.csv'),
    'Variance' : pd.read_csv(folder + 'data_Variance.csv'),
    'Mean' : pd.read_csv(folder + 'data_Mean.csv')
    }
#%% Load labels
labels_folder = 'data/labels/'
cooler_labels = np.loadtxt(labels_folder + 'labels_cooler.txt')
valve_labels = np.loadtxt(labels_folder + 'labels_valve.txt')
pump_labels = np.loadtxt(labels_folder + 'labels_pump.txt')
accumulator_labels = np.loadtxt(labels_folder + 'labels_accumulator.txt')
stableFlag_labels = np.loadtxt(labels_folder + 'labels_stableFlag.txt')
#%% Sensor names list
all_sensors = ['Temperature sensor 1', #0
               'Temperature sensor 2', #1
               'Temperature sensor 3', #2
               'Temperature sensor 4', #3
               'Vibration sensor', #4
               'Cooling efficiency', #5
               'Cooling power', #6
               'Efficiency factor', #7
               'Flow sensor 1', #8
               'Flow sensor 2', #9
               'Pressure sensor 1', #10
               'Pressure sensor 2', #11
               'Pressure sensor 3', #12
               'Pressure sensor 4', #13
               'Pressure sensor 5', #14
               'Pressure sensor 6', #15
               'Motor power'] #16
#%%
sensors_idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#%% COOLER
#Params
model = RandomForestClassifier(n_estimators=100)
TimeParam_df = TimeParams['RMS']

good_idxs = []
old_max_acc = 0
max_acc = 0
while  max_acc >= old_max_acc:
    old_max_acc = max_acc
    combinations_list = get_combinations(good_idxs)
    acc_list = []
    for sensors_combination in combinations_list:
        Model = model
        sensor_list = [all_sensors[i] for i in sensors_combination]
        X_train,X_test,Y_train,Y_test = get_XY_TrainTest(TimeParam_df[sensor_list],
                                                          cooler_labels)
        Model.fit(X_train, Y_train)
        acc_list.append(accuracy_score(Y_test,Model.predict(X_test)))    
    max_acc = np.max(np.array(acc_list))   
    max_idxs = get_max_idxs(acc_list, max_acc)
    sensors_idx_list.remove(max_idxs[0])
    good_idxs.append(max_idxs[0])
print('Max accuracy: ',max_acc)
print('Mejores sensores: ',sorted(good_idxs))





