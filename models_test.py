# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:52:27 2021

@author: ihuer
"""
#%% IMPORTS
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from utils.utils_forward_select import conditions_fwd_select
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
    'Flow sensor 2' : np.loadtxt(folder + "FS2.txt"),
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
#%% MODELS
KNN_models = {
    'KNN 1' : KNeighborsClassifier(n_neighbors=1),
    'KNN 2' : KNeighborsClassifier(n_neighbors=2),
    'KNN 3' : KNeighborsClassifier(n_neighbors=3),
    'KNN 4' : KNeighborsClassifier(n_neighbors=4),
    'KNN 5' : KNeighborsClassifier(n_neighbors=5),
    'KNN 6' : KNeighborsClassifier(n_neighbors=6)
    }

RF_models = {
    'RF 40' : RandomForestClassifier(n_estimators=40),
    'RF 60' : RandomForestClassifier(n_estimators=60),
    'RF 80' : RandomForestClassifier(n_estimators=80),
    'RF 100' : RandomForestClassifier(n_estimators=100),
    'RF 120' : RandomForestClassifier(n_estimators=120)
    }

SVM_models = {
    'SVM rbf' : SVC(kernel='rbf'), 
    'SVM linear' : SVC(kernel='linear'),
    'SVM sigmoid' : SVC(kernel='sigmoid')     
    }

LDA_models = {'LDA 1' : LinearDiscriminantAnalysis(n_components=1)}

TimeParams_list = [
    'RMS'
    'P2P',
    'Variance',
    'Mean'
    ]
#%% TIME WINDOWS
time_windows_params = [
    'win60_olap0', # 1 per instance 
    'win30_olap0', # 2 per instance
    'win20_olap0', # 3 per instance
    'win22_olap10',# 4 per instance
    'win20_olap10',# 5 per instance
    'win18_olap10',# 6 per instance
    'win15_olap8',# 7 per instance
    ]
#%% RUN MODELS
for win_olap_str in time_windows_params:
    _ = conditions_fwd_select(Raw_data, conditions, SVM_models, TimeParams_list,
                              win_olap_str)
#%%
