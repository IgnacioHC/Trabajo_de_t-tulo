# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import numpy as np
import pandas as pd
import json

from sklearn.svm import SVC
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
#%% load_SVMconditon_MaxAcc_idxs
def load_SVMconditon_MaxAcc_idxs(win_olap_str, condition):
    head_path = 'results/max_accuracy_idxs/'  + win_olap_str + '/max_idxs_'
    file_path = head_path + 'SVM_' + win_olap_str + '.json'
    data = json.load(open(file_path,))
    return data[condition]
#%% get_sets
def get_sets(win_olap_str, TimeParam, condition, Kernel, MaxAcc_idxs):
    win = int(win_olap_str.split('_')[0][3:])
    olap = int(win_olap_str.split('_')[0][4:])
    sets = preprocess_data(Raw_data, conditions[condition], TimeParam,
                           win, olap)
    TimeParam_Ker_AccIdxs = MaxAcc_idxs['SVM ' + Kernel][TimeParam]
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
    TimeParam_Ker_AccIdxs.sort()
    sensor_list = [sensor_names[i] for i in TimeParam_Ker_AccIdxs]
    X_train, X_test = sets[0][sensor_list].values, sets[1][sensor_list].values
    Y_train, Y_test = sets[2], sets[3]
    return X_train, X_test, Y_train, Y_test
#%% GridSearc
def GridSearch(Kernel, win_olap_str, TimeParam, condition, MaxAcc_idxs,
               param_grid):
    X_train, X_test, Y_train, Y_test = get_sets(win_olap_str, TimeParam,
                                                condition, Kernel,
                                                MaxAcc_idxs)
    TimeParam_accs = []
    for c in param_grid['C']:
        for Gamma in param_grid['gamma']:
            model = SVC(kernel=Kernel, C=c, gamma=Gamma).fit(X_train, Y_train)
            acc = accuracy_score(Y_test, model.predict(X_test))
            TimeParam_accs.append(acc)
    return TimeParam_accs
#%% TimeParam_GridSearch
def TimeParam_GridSearch(Kernel, win_olap_str, TimeParam_list, condition,
                         MaxAcc_idxs, param_grid, save = False):
    accuracies_dict = {}
    for TimeParam in TimeParam_list:
        print(TimeParam)
        TimeParam_accs = GridSearch(Kernel, win_olap_str, TimeParam, condition,
                                    MaxAcc_idxs, param_grid)
        accuracies_dict[TimeParam] = TimeParam_accs
    accs_df = pd.DataFrame.from_dict(accuracies_dict)
    gamma = param_grid['gamma']*len(param_grid['C'])
    C = []
    for c in param_grid['C']:
        C = C + [c]*len(param_grid['gamma'])
    accs_df['gamma'] = gamma
    accs_df['C'] = C
    if save == True:
        head = 'results/accuracies/' + condition + '/' + win_olap_str + '/'
        tail1 = condition + '_' + 'SVM' + Kernel + 'Params_' 
        tail2 = win_olap_str + '.csv'
        file_path = head + tail1 + tail2
        accs_df.to_csv(file_path)
    else:
        pass
    return accs_df
#%%
conditions_list = [
    'cooler',
    'stableFlag'
    ]

time_windows = [
    'win60_olap0', # 1 per instance 
    'win30_olap0', # 2 per instance
    'win20_olap0', # 3 per instance
    'win22_olap10',# 4 per instance
    'win20_olap10',# 5 per instance
    'win18_olap10',# 6 per instance
    ]

TimeParam_list = ['RMS', 'P2P', 'Mean', 'Variance']

param_grid = {'C': [10, 100, 1000, 10**4, 10**5, 10**6, 10**7],
              'gamma': [0.1, 1, 10, 100]}



for condition in conditions_list:
    print('======', condition, '======')
    MaxAcc_idxs = load_SVMconditon_MaxAcc_idxs('win60_olap0', condition)
    accs_df = TimeParam_GridSearch('rbf', 'win60_olap0', TimeParam_list,
                                   condition, MaxAcc_idxs, param_grid,
                                   save = True)














    
    


#%%%
# for condition in conditions_list:
#     for win_olap_str in time_windows:
#         condition_accs = load_SVMconditon_MaxAcc_idxs(win_olap_str, condition)
#         for Kernel in ['rbf', 'linear', 'sigmoid']:
#             accs_df = TimeParam_GridSearch(Kernel, win_olap_str,
#                                            TimeParam_list, condition,
#                                            condition_accs, save = True)

