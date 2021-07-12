# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils_ForwardSelect import get_TimeParams_df_dict
#%% load_condition_accuracies
def load_condition_accuracies(cond_accuracies_path):
    """
    Carga las accuracies correspondientes a la condicion entregada
    --------------------------------------------------------------------------
    Parameters     

    cond_accuracies_path: string
    
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    condition_accuracies = {}
    windows_folders_list = os.listdir(cond_accuracies_path)
    for window in windows_folders_list:
        window_accuracies = {}
        models_CSVs_list = os.listdir(cond_accuracies_path + window)
        for model_accuracies_csv in models_CSVs_list:
            models_name = model_accuracies_csv.split('_')[1]
            models_path = cond_accuracies_path + window + '/' + model_accuracies_csv
            window_accuracies[models_name] = pd.read_csv(models_path)
        condition_accuracies[window] = window_accuracies
    return condition_accuracies

#%% get_len_timeWindow
def get_len_timeWindow(win_olap_str, train_sz = 0.7):
    """
    Retorna el largo de la ventana temporal usada en el entrenamiento
    --------------------------------------------------------------------------
    Parameters     

    win_olap_str: string
    
    train_sz: float, 0 <= train_sz <= 1
    
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    win = int(win_olap_str.split('_')[0][3:])
    olap = int(win_olap_str.split('_')[1][4:])
    train_instances = np.floor(2205 * train_sz).astype(int)
    len_per_instance = train_instances * int(np.floor((60-olap)/(win-olap)))
    return len_per_instance
#%% get_conditions_data_dict
def get_conditions_data_dict(labels_dict, TimeParams_list, RawData_dict,
                             t_win, t_olap):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    conditions_data_dict = {}
    for condition_name, condition_labels in labels_dict.items():
        TimeParams_df_dict = get_TimeParams_df_dict(TimeParams_list,
                                                    RawData_dict,
                                                    condition_labels, t_win,
                                                    t_olap, train_sz = 0.7,
                                                    random_st = 19)
    conditions_data_dict[condition_name] = TimeParams_df_dict
    return conditions_data_dict
#%% plot_RF_accuracies
def plot_RF_accuracies(condition_accuracies, TimeParams_list):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    plt.figure(figsize = (12,10), dpi=80)
    N_estimators = np.array([40, 60, 80, 100, 120])
    # Iter over conditions
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        # Iter over time windows
        for win_olap_str in condition_accuracies.keys():
            accuracies = condition_accuracies[win_olap_str]['RF'][TimeParam].to_numpy()
            plt.plot(N_estimators, accuracies, label = win_olap_str)
        plt.title(TimeParam, size = 12)
        plt.xlabel('Cantidad de árboles', size=10)
        plt.ylabel('Accuracy', size=10)
        plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('RF_accuracies.png')
#%%

#%% plot_KNN_accuracies
def plot_KNN_accuracies(condition_accuracies, TimeParams_list):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    plt.figure(figsize = (8,4), dpi = 80)
    N_neighbors = np.array([1, 2, 3, 4, 5, 6])
    # Iter over conditions
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        # Iter over time windows
        for win_olap_str in condition_accuracies.keys():
            accuracies = condition_accuracies[win_olap_str]['KNN'][TimeParam].to_numpy()
            plt.plot(N_neighbors, accuracies, label = win_olap_str)
        #Fig text
        plt.title(TimeParam, size = 12)
        plt.xlabel('Cantidad de vecinos', size=10)
        plt.ylabel('Accuracy', size=10)
        plt.legend()
    plt.tight_layout()
    plt.show()
#%%
def plot_SVM_accuracies(condition_accuracies, TimeParams_list):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    plt.figure(figsize = (12,10), dpi=80)
    # Iter over conditions
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        # Iter over trees
        kernels = ['rbf', 'linear', 'sigmoid']
        for kernel_idx in range(3):
            model_SVM_name = 'SVM_' + kernels[kernel_idx]
            accuracies = []
            win_lens = [] 
            for win_olap_str in condition_accuracies.keys():
                accuracies_df = condition_accuracies[win_olap_str]['SVM']
                accuracies.append(accuracies_df.iloc[kernel_idx][TimeParam])
                win_lens.append(get_len_timeWindow(win_olap_str))
            plt.plot(win_lens, accuracies, label = model_SVM_name)
        plt.title(TimeParam, size = 12)
        plt.xlabel('Largo ventana de entreanamiento', size=10)
        plt.ylabel('Accuracy', size=10)
        plt.legend()
    plt.tight_layout()
    plt.show()
#%%
def plot_model_accuracies(condition_accuracies, TimeParams_list,
                          fig_sz = (12,9)):
    """
    Plotea las accuracies de los distintos modelos en funcion de el largo de
    la ventana de entrenamiento.
    """
    plt.figure(figsize = fig_sz, dpi = 200)
    #Iterate over Time params
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        for win_olap_str, window_data in condition_accuracies.items():
            len_win = get_len_timeWindow(win_olap_str)
            for models_name, model_df in window_data.items():
                x = np.array([len_win] * len(model_df))
                accuracies = model_df[TimeParam]
                plt.scatter(x, accuracies)
        #Fig text
        plt.title(TimeParam, size = 10)
        plt.xlabel('Length of time window used for training', size=8)
        plt.ylabel('Model accuracy', size=8)
    
    plt.tight_layout()
    #plt.savefig('test_fig.png')
    plt.show()