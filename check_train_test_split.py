# -*- coding: utf-8 -*-
"""
@author: Ignacio
"""
import numpy as np
import pandas as pd

from utils.utils_preprocessing import get_TimeParam_dict

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%% Load raw data
folder = 'data/data_raw/'
dataRaw_dict = {
    'Temperature sensor 1' : np.loadtxt(folder + "TS1.txt"),
    'Temperature sensor 2' : np.loadtxt(folder + "TS2.txt"),
    'Temperature sensor 3' : np.loadtxt(folder + "TS3.txt"),
    'Temperature sensor 4' : np.loadtxt(folder + "TS4.txt"),
    'Vibration sensor' : np.loadtxt(folder + "VS1.txt"),
    'Cooling efficiency' : np.loadtxt(folder + "CE.txt"),
    'Cooling power' : np.loadtxt(folder + "CP.txt"),
    'Efficiency factor' : np.loadtxt(folder + "SE.txt"),
    'Flow sensor 1' : np.loadtxt(folder + "FS1.txt"),
    'Flow sensor 2' : np.loadtxt(folder + "FS2.txt"),
    'Pressure sensor 1' : np.loadtxt(folder + "PS1.txt"),
    'Pressure sensor 2' : np.loadtxt(folder + "PS2.txt"),
    'Pressure sensor 3' : np.loadtxt(folder + "PS3.txt"),
    'Pressure sensor 4' : np.loadtxt(folder + "PS4.txt"),
    'Pressure sensor 5' : np.loadtxt(folder + "PS5.txt"),
    'Pressure sensor 6' : np.loadtxt(folder + "PS6.txt"),
    'Motor power' : np.loadtxt(folder + "EPS1.txt")
}
#%%
Y = np.loadtxt('data/labels/labels_pump.txt')
X = dataRaw_dict['Temperature sensor 1']
sets = train_test_split(X,Y, train_size=0.7, random_state=19)
#%%
sensor_sets =  {'Temperature sensor 2':train_test_split(dataRaw_dict['Temperature sensor 2'],Y,train_size=0.7, random_state=19),
                'Temperature sensor 3':train_test_split(dataRaw_dict['Temperature sensor 3'],Y,train_size=0.7, random_state=19),
                'Temperature sensor 4':train_test_split(dataRaw_dict['Temperature sensor 4'],Y,train_size=0.7, random_state=19),
                'Vibration sensor':train_test_split(dataRaw_dict['Vibration sensor'],Y,train_size=0.7, random_state=19),
                'Cooling efficiency':train_test_split(dataRaw_dict['Cooling efficiency'],Y,train_size=0.7, random_state=19),
                'Cooling power':train_test_split(dataRaw_dict['Cooling efficiency'],Y,train_size=0.7, random_state=19),
                'Efficiency factor':train_test_split(dataRaw_dict['Cooling efficiency'],Y,train_size=0.7, random_state=19),
                'Flow sensor 1':train_test_split(dataRaw_dict['Flow sensor 1'],Y,train_size=0.7, random_state=19),
                'Flow sensor 2':train_test_split(dataRaw_dict['Flow sensor 2'],Y,train_size=0.7, random_state=19),
                'Motor power':train_test_split(dataRaw_dict['Motor power'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 1':train_test_split(dataRaw_dict['Pressure sensor 1'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 2':train_test_split(dataRaw_dict['Pressure sensor 2'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 3':train_test_split(dataRaw_dict['Pressure sensor 3'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 4':train_test_split(dataRaw_dict['Pressure sensor 4'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 5':train_test_split(dataRaw_dict['Pressure sensor 5'],Y,train_size=0.7, random_state=19),
                'Pressure sensor 6':train_test_split(dataRaw_dict['Pressure sensor 6'],Y,train_size=0.7, random_state=19)}
#%%
def all_equal(array1, array2):
    lst = []
    for i in range(60):
        if array1[i] == array2[i]:
            lst.append(1)
        else:
            lst.append(0)
    if np.mean(np.array(lst)) == 1.0:
        return True
    else:
        return False
#%%
idxs = []
for i in range(sets[0].shape[0]):
    instance = sets[0][i]
    for j in range(2205):
        ins = X[j]
        if all_equal(instance, ins) == True:
            idxs.append((i,j))
        else:
            pass
#%%
for sensor_name, sensor_data in sensor_sets.items():
    train_set = sensor_data[0]
    for i in range(1543):
        arr1 = dataRaw_dict[sensor_name][idxs[i][1]]
        arr2 = train_set[i]
        random = np.random.randint(0, train_set.shape[1])
        print(arr1[random], ' ', arr2[random])
#%%






