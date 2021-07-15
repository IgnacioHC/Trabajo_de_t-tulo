# -*- coding: utf-8 -*-
"""
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
def get_len_training(win_olap_str, train_sz = 0.7):
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
    len_training = train_instances * int(np.floor((60-olap)/(win-olap)))
    return len_training
#%% plot_RF_accuracies
def plot_RF_accuracies(condition_accs, TimeParams_list):
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
        for win_olap_str in condition_accs.keys():
            #Load accuracies
            accs = condition_accs[win_olap_str]['RF'][TimeParam].to_numpy()
            len_training = get_len_training(win_olap_str)
            lbl = 'Datos de entrenamiento : {}'.format(len_training)
            #Plots
            plt.plot(N_estimators, accs, label = lbl)
        plt.title(TimeParam, size = 12)
        plt.xlabel('Cantidad de árboles', size=10)
        plt.ylabel('Accuracy', size=10)
        plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('RF_accuracies.png')
#%%
def plot_RF_GiniEntro_accs(condition_accs, TimeParams_list, time_windows,
                           fig_sz = (12,10), tit_sz = 13, ax_sz  = 12):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    windows_colors = {
        'win60_olap0' : '#2b20bd',  
        'win30_olap0' : '#0acf1b',  
        'win22_olap10' : '#b12fbd',
        'win15_olap8' : '#c9783e',       
        }
    plt.figure(figsize = fig_sz, dpi=100)
    N_estimators = np.array([40, 60, 80, 100, 120])
    # Iter over conditions
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        # Iter over time windows
        for win_olap_str in time_windows:
            #Load accuracies
            accs_gini = condition_accs[win_olap_str]['RF'][TimeParam].to_numpy()
            accs_entro = condition_accs[win_olap_str]['RFentropy'][TimeParam].to_numpy()
            #Plots
            len_training = get_len_training(win_olap_str)
            lbl = 'Datos de entrenamiento : {}'.format(len_training)
            plt.plot(N_estimators, accs_gini,
                     color = windows_colors[win_olap_str], marker = 'o',
                     label = lbl)
            plt.plot(N_estimators, accs_entro,
                     color = windows_colors[win_olap_str], marker = 's',
                     label = lbl)
        plt.title(TimeParam, size = tit_sz )
        plt.xlabel('Cantidad de árboles', size = ax_sz)
        plt.ylabel('Accuracy', size = ax_sz)
        plt.legend()
    plt.tight_layout()
    plt.show()
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
def plot_KNN_UniDist_accs(condition_accs, TimeParams_list, time_windows,
                           fig_sz = (12,10), tit_sz = 13, ax_sz  = 12):
    """
    --------------------------------------------------------------------------
    Parameters     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    windows_colors = {
        'win60_olap0' : '#2b20bd',  
        'win30_olap0' : '#0acf1b',  
        'win22_olap10' : '#b12fbd',
        'win15_olap8' : '#c9783e',       
        }
    plt.figure(figsize = fig_sz, dpi=100)
    N_neighbors = np.array([1, 2, 3, 4, 5, 6])
    # Iter over conditions
    for TimeParam, i in zip(TimeParams_list, range(len(TimeParams_list))):
        plt.subplot(np.ceil(len(TimeParams_list)/2).astype(int), 2, i+1)
        # Iter over time windows
        for win_olap_str in time_windows:
            #Load accuracies
            accs_uni = condition_accs[win_olap_str]['KNN'][TimeParam].to_numpy()
            accs_dist = condition_accs[win_olap_str]['KNNdistance'][TimeParam].to_numpy()
            #Plots
            len_training = get_len_training(win_olap_str)
            lbl = 'Datos de entrenamiento : {}'.format(len_training)
            plt.plot(N_neighbors, accs_uni,
                     color = windows_colors[win_olap_str], marker = 'o',
                     label = lbl)
            plt.plot(N_neighbors, accs_dist,
                     color = windows_colors[win_olap_str], marker = 's',
                     label = lbl)
        plt.title(TimeParam, size = tit_sz )
        plt.xlabel('Cantidad de vecinos', size = ax_sz)
        plt.ylabel('Accuracy', size = ax_sz)
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
                win_lens.append(get_len_training(win_olap_str))
            plt.scatter(win_lens, accuracies, label = model_SVM_name)
        plt.title(TimeParam, size = 12)
        plt.xlabel('Largo ventana de entreanamiento', size=10)
        plt.ylabel('Accuracy', size=10)
        plt.legend()
    plt.tight_layout()
    plt.show()
#%%
#def plot_SVM_HeatMap():
    
    
    
    