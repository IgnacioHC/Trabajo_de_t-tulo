# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:53:03 2021

@author: ihuer
"""
#%% IMPORTS
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% load_condition_accuracies
def load_condition_accuracies(cond_accuracies_path):
    """
    Carga las accuracies correspondientes a la condicion entregada
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
#Test
#cooler_accuracies = load_condition_accuracies('results/accuracies/cooler/')
#%% get_len_timeWindow
def get_len_timeWindow(win_olap_str, train_sz = 0.7):
    """
    Retorna el largo de la ventana temporal usada en el entrenamiento
    """
    win, olap = int(win_olap_str.split('_')[0][3:]), int(win_olap_str.split('_')[1][4:])
    train_instances = np.floor(2205 * train_sz).astype(int)
    len_per_instance = train_instances * int(np.floor((60-olap)/(win-olap)))
    return len_per_instance
#test
#timeWindow = get_len_timeWindow('win20_olap5')
#%% plot_model_accuracies
def plot_model_accuracies(condition_accuracies, fig_sz = (12,9)):
    """
    Plotea las accuracies de los distintos modelos en funcion de el largo de
    la ventana de entrenamiento.
    """
    plt.figure(figsize = fig_sz, dpi = 200)
    #Iterate over Time params
    TimeParams_list = ['RMS', 'P2P', 'Variance', 'Mean']
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(2, 2, i+1)
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
    plt.savefig('test_fig.png')
    plt.show()
#%%
condition_accuracies = load_condition_accuracies('results/accuracies/valve/')
condition_accuracies = {'win20_olap5':condition_accuracies['win20_olap5'],
                        'win15_olap5':condition_accuracies['win15_olap5'],
                        'win10_olap5':condition_accuracies['win10_olap5']}
        
plot_model_accuracies(condition_accuracies)

