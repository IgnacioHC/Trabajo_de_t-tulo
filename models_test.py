# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:52:27 2021

@author: ihuer
"""
#%% IMPORTS
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.utils_forward_selection import conditions_fwd_select
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
    'stable_flag' : np.loadtxt(labels_folder + 'labels_stableFlag.txt')
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

LDA_models = {
    'LDA 1' : LinearDiscriminantAnalysis(n_components=1)
    #'LDA 2' : LinearDiscriminantAnalysis(n_components=2),
    #'LDA 3' : LinearDiscriminantAnalysis(n_components=3)
    }

TimeParams_list = [
    'RMS',
    'P2P',
    'Variance',
    'Mean'
    ]
#%% RUN MODELS
_ = conditions_fwd_select(Raw_data, conditions, LDA_models, TimeParams_list, 'win20_olap0')
_ = conditions_fwd_select(Raw_data, conditions, LDA_models, TimeParams_list, 'win20_olap5')
_ = conditions_fwd_select(Raw_data, conditions, LDA_models, TimeParams_list, 'win15_olap5')
_ = conditions_fwd_select(Raw_data, conditions, LDA_models, TimeParams_list, 'win10_olap5')