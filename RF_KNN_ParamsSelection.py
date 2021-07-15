# -*- coding: utf-8 -*-
"""
@author: Ignacio
"""
#%% IMPORTS
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
def load_condition_MaxAcc_idxs(win_olap_str, model_name, condition):
    """
    Carga 
    --------------------------------------------------------------------------
    Parameters
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    
    model_name: string
    
    condition: string
        Name of the condition
    --------------------------------------------------------------------------
    Returns
    out: dictionary
        Dictionary in the form {model_name:{
                                    TimeParam_name : MaxAcc_idxs_list
                                        }
                                    }
    """
    if model_name == 'RFentropy':
        model_name = 'RF'
    elif model_name == 'KNNdistance':
        model_name = 'KNN'
    else:
        pass
    head_path = 'results/max_accuracy_idxs/'  + win_olap_str + '/max_idxs_'
    file_path = head_path + model_name + '_' + win_olap_str + '.json'
    data = json.load(open(file_path,))
    return data[condition]
#%% set_SubModel
def set_SubModel(SubModel_name):
    if SubModel_name[:2] == 'RF':
        n_trees = int(SubModel_name[3:])
        SubModel = RandomForestClassifier(n_estimators = n_trees,
                                          criterion = 'entropy')
    elif SubModel_name[:3] == 'KNN':
        N_neighbors = int(SubModel_name[4:])
        SubModel = KNeighborsClassifier(n_neighbors = N_neighbors,
                                        weights = 'distance')
    else:
        SubModel = None
    return SubModel
#%% train_SubModels
def train_SubModels(SubModel_name, SubModel_idxs, sets):
    sensor_names = ['Temperature sensor 1', #0
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
    SubModel_idxs.sort()
    sensor_list = [sensor_names[i] for i in SubModel_idxs]
    X_train, X_test = sets[0][sensor_list].values, sets[1][sensor_list].values
    Y_train, Y_test = sets[2], sets[3]
    SubModel = set_SubModel(SubModel_name)
    SubModel.fit(X_train, Y_train)
    return accuracy_score(Y_test, SubModel.predict(X_test))
#%% get_ModelAccuracies
def get_ModelAccuracies(Raw_data, win_olap_str, condition, conditions_dict,
                        TimeParam, model_name):
    # Get train and test sets
    win = int(win_olap_str.split('_')[0][3:])
    olap = int(win_olap_str.split('_')[0][4:])
    sets = preprocess_data(Raw_data, conditions_dict[condition], TimeParam,
                           win, olap)
    MaxAcc_idxs = load_condition_MaxAcc_idxs(win_olap_str, model_name, condition)
    # Iter over SubModels
    SubModels_accuracies = {}
    for SubModel_name, SubModel_MaxAcc_idxs in MaxAcc_idxs.items():
        # Get the corresponding MaxAcc idxs for the TimeParam and SubModel
        SubModel_idxs = SubModel_MaxAcc_idxs[TimeParam]
        # Get and save accuracies
        accuracy = train_SubModels(SubModel_name, SubModel_idxs, sets)
        SubModels_accuracies[SubModel_name] = accuracy
    return SubModels_accuracies
#%%
def save_accs_df(accs_df, win_olap_str, condition, model_name):
    head = 'results/accuracies/' + condition + '/' + win_olap_str + '/'
    tail = condition + '_' + model_name + '_' + win_olap_str + '.csv'
    file_path = head + tail
    accs_df.to_csv(file_path)
#%%
def get_accs_df(win_olap_str, condition, conditions_dict, model_name,
                save = False):
    TimeParam_accuracies = {
        'RMS' : get_ModelAccuracies(Raw_data, win_olap_str, condition, 'RMS',
                                    model_name),
        'P2P' : get_ModelAccuracies(Raw_data, win_olap_str, condition, 'P2P',
                                    model_name),
        'Mean' : get_ModelAccuracies(Raw_data, win_olap_str, condition, 'Mean',
                                      model_name),
        'Variance' : get_ModelAccuracies(Raw_data, win_olap_str, condition,
                                          'Variance', model_name)
        }
    SubModel_series_list = []
    idx = []
    for SubModel_name in list(TimeParam_accuracies['RMS'].keys()):
        row = {}
        for TimeParam, SubModels_accuracies in TimeParam_accuracies.items():
            row[TimeParam] = SubModels_accuracies[SubModel_name]
        SubModel_series_list.append(pd.Series(row))
        idx.append(SubModel_name)
    accs_df = pd.DataFrame(SubModel_series_list, index = idx)
    if save == True:
        save_accs_df(accs_df, win_olap_str, condition, model_name)
    else:
        pass
    return accs_df 
#%%
time_windows = [
    'win60_olap0', # 1 per instance 
    'win30_olap0', # 2 per instance
    'win20_olap0', # 3 per instance
    'win22_olap10',# 4 per instance
    'win20_olap10',# 5 per instance
    'win18_olap10',# 6 per instance
    'win15_olap8',# 7 per instance
    ]

models = [
    'RFentropy',
    'KNNdistance'
    ]

for model_name in models:
    for condition in conditions.keys():
        for win_olap_str in time_windows:
            accs_df = get_accs_df(win_olap_str, condition, model_name,
                                  save = True)
#%% 

