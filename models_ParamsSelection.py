# -*- coding: utf-8 -*-
"""
@author: Ignacio
"""
#%% IMPORTS
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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
