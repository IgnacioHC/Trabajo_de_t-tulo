# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:11:30 2021

@author: ihuer
"""
#%% Imports
import pandas as pd
import numpy as np
from utils.utils_forward_selection import forward_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#%% Load data
folder = 'data/data_TimeParams/'
RMS_df = pd.read_csv(folder + 'data_RMS.csv')
P2P_df = pd.read_csv(folder + 'data_P2P.csv')
Var_df = pd.read_csv(folder + 'data_Variance.csv')
Mean_df = pd.read_csv(folder + 'data_Mean.csv')
#%% Load labels
labels_folder = 'data/labels/'
cooler_labels = np.loadtxt(labels_folder + 'labels_cooler.txt')
valve_labels = np.loadtxt(labels_folder + 'labels_valve.txt')
pump_labels = np.loadtxt(labels_folder + 'labels_pump.txt')
accumulator_labels = np.loadtxt(labels_folder + 'labels_accumulator.txt')
stableFlag_labels = np.loadtxt(labels_folder + 'labels_stableFlag.txt')
#%%
forward_selection(RMS_df,RandomForestClassifier(n_estimators=100),valve_labels)