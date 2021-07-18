# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
TimeParms_ES ={
    'RMS' : 'RMS',
    'P2P' : 'Valor peak to peak',
    'Mean' : 'Media',
    'Variance' : 'Varianza'
    }
#%%
fig_width = 10
#%% load_condition_accuracies
def load_condition_accuracies(cond_accuracies_path):
    """
    Carga las accuracies correspondientes a la condicion (clasificasión)
    entregada, para todas las ventanas de tiempo y todos los parámetros de
    tiempo.
    --------------------------------------------------------------------------
    Parameters     

    cond_accuracies_path: string
        cond_accuracies_path = ''results/accuracies/condition/'     
    -------------------------------------------------------------------------
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
#%% get_len_PerInst
def get_len_PerInst(win_olap_str):
    """
    Retorna la cantidad de datos extraidos por el parámetro de tiempo en cada
    ciclo de operación.
    --------------------------------------------------------------------------
    Parameters     
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    win = int(win_olap_str.split('_')[0][3:])
    olap = int(win_olap_str.split('_')[1][4:])
    return int(np.floor((60-olap)/(win-olap)))
#%%
def plot_RF_GiniEntro_accs(condition, TimeParams_list, fig_sz = (fig_width,12), 
                          subplt_tit_sz = 12, subplt_XYlabel_sz = 10,
                          shareY = True, legend_loc = (1.3, 0.6)):
    """
    --------------------------------------------------------------------------
    Parameters
    
    condition: str
        condition name.
    
    TimeParams_list: list
        List containing the names of the time parameter to be ploted.
        example: ['RMS', 'Mean']
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    
    fig_sz: tuple, default = ()
        Figure's size.
    
    subplt_tit_sz: float or int, deafult =
        Subplot title font title size.
    
    subplt_XYlabel_sz: float or int, deafult =
        Subplot xlabel and ylabel font size.
    
    shareY: bool, default = True
        If true, subplots share the Y axis scale.
    
    legend_loc: (float, float), default = ()
        Location for the figure legend.
    --------------------------------------------------------------------------
    Returns
    out: plots
    """
    time_windows_colors = {
        'win60_olap0' : '#2b20bd',  
        'win30_olap0' : '#0acf1b',  
        'win22_olap10' : '#b12fbd',
        'win15_olap8' : '#c9783e',
        }
    path = 'results/accuracies/' + condition + '/'
    condition_accs = load_condition_accuracies(path)
    N_estimators = np.array([40, 60, 80, 100, 120])
    cols, rows = 2, np.ceil(len(TimeParams_list)/2).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize = fig_sz, dpi = 100,
                            sharey = shareY)
    #fig.suptitle('Accuracies')
    for TimeParam, ax in zip(TimeParams_list, fig.axes):
        curves = []
        for win_olap_str in time_windows_colors.keys():
            accs_gini = condition_accs[win_olap_str]['RF'][TimeParam].to_numpy()
            accs_entro = condition_accs[win_olap_str]['RFentropy'][TimeParam].to_numpy()
            #Plots
            curv1 = ax.plot(N_estimators, accs_gini, marker = 'o',
                            color = time_windows_colors[win_olap_str])
            curv2 = ax.plot(N_estimators, accs_entro, marker = 's',
                            color = time_windows_colors[win_olap_str])
            curves = curves + [curv1 + curv2]
            #Subplot Text
            ax.set_title(TimeParms_ES[TimeParam], size = subplt_tit_sz )
            ax.set_xlabel('Cantidad de árboles', size = subplt_XYlabel_sz)
            ax.set_ylabel('Accuracy', size = subplt_XYlabel_sz)
    lbls = ['Criterio gini,\n1 dato por ciclo',
            'Criterio de entropía,\n1 dato por ciclo',
            'Criterio gini,\n 2 datos por ciclo',
            'Criterio de entropía,\n2 datos por ciclo',
            'Peso uniforme,\n4 datos por ciclo',
            'Criterio de entropía,\n4 datos por ciclo',
            'Criterio gini,\n7 datos por ciclo',
            'Criterio de entropía,\n7 datos por ciclo']
    fig.legend(curves, labels = lbls, bbox_to_anchor = legend_loc, ncol = 1,
               fancybox=True, fontsize = 'medium')
    plt.tight_layout()
    plt.show()
#%%
def plot_KNN_UniDist_accs(condition, TimeParams_list, fig_sz = (fig_width,12), 
                          subplt_tit_sz = 12, subplt_XYlabel_sz = 10,
                          shareY = True, legend_loc = (1.3, 0.6)):
    """
    --------------------------------------------------------------------------
    Parameters
    
    condition: str
        condition name.
    
    TimeParams_list: list
        List containing the names of the time parameter to be ploted.
        example: ['RMS', 'Mean']
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    
    fig_sz: tuple, default = ()
        Figure's size.
    
    subplt_tit_sz: float or int, deafult =
        Subplot title font title size.
    
    subplt_XYlabel_sz: float or int, deafult =
        Subplot xlabel and ylabel font size.
    
    shareY: bool, default = True
        If true, subplots share the Y axis scale.
    
    legend_loc: (float, float), default = ()
        Location for the figure legend.
    --------------------------------------------------------------------------
    Returns
    out: plots
    """
    time_windows_colors = {
        'win60_olap0' : '#2b20bd',  
        'win30_olap0' : '#0acf1b',  
        'win22_olap10' : '#b12fbd',
        'win15_olap8' : '#c9783e',
        }
    path = 'results/accuracies/' + condition + '/'
    condition_accs = load_condition_accuracies(path)
    N_neighbors = np.array([1, 2, 3, 4, 5, 6])
    cols, rows = 2, np.ceil(len(TimeParams_list)/2).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize = fig_sz, dpi = 100,
                            sharey = shareY)
    #fig.suptitle('Accuracies')
    for TimeParam, ax in zip(TimeParams_list, fig.axes):
        curves = []
        for win_olap_str in time_windows_colors.keys():
            accs_uni = condition_accs[win_olap_str]['KNN'][TimeParam].to_numpy()
            accs_dist = condition_accs[win_olap_str]['KNNdistance'][TimeParam].to_numpy()
            #Plots
            curv1 = ax.plot(N_neighbors, accs_uni, marker = 'o',
                            color = time_windows_colors[win_olap_str])
            curv2 = ax.plot(N_neighbors, accs_dist, marker = 's',
                            color = time_windows_colors[win_olap_str])
            curves = curves + [curv1 + curv2]
            #Subplot Text
            ax.set_title(TimeParms_ES[TimeParam], size = subplt_tit_sz )
            ax.set_xlabel('Cantidad de vecinos', size = subplt_XYlabel_sz)
            ax.set_ylabel('Accuracy', size = subplt_XYlabel_sz)
    lbls = ['Peso uniforme,\n1 dato por ciclo',
            'Peso basado en la distancia,\n1 dato por ciclo',
            'Peso uniforme,\n2 datos por ciclo',
            'Peso basado en la distancia,\n2 datos por ciclo',
            'Peso uniforme,\n4 datos por ciclo',
            'Peso basado en la distancia,\n4 datos por ciclo',
            'Peso uniforme,\n7 datos por ciclo',
            'Peso basado en la distancia,\n7 datos por ciclo']    
    fig.legend(curves, labels = lbls, bbox_to_anchor = legend_loc, ncol = 1,
               fancybox=True, fontsize = 'medium')
    plt.tight_layout()
    plt.show()
#%%
def plot_SVM_accuracies(condition, TimeParams_list, fig_sz = (fig_width,8),
                        shareY = True, subplt_tit_sz = 12,
                        subplt_XYlabel_sz  = 10, legend_loc = 'upper right'):
    """
    --------------------------------------------------------------------------
    Parameters
    
    condition: str
        condition name.
    
    TimeParams_list: list
        List containing the names of the time parameter to be ploted.
        example: ['RMS', 'Mean']
    
    fig_sz: tuple, default = ()
        Figure's size.
    
    subplt_tit_sz: float or int, deafult =
        Subplot title font title size.
    
    subplt_XYlabel_sz: float or int, deafult =
        Subplot xlabel and ylabel font size.
    
    legend_loc: (float, float), default = ()
        Location for the figure legend.
    --------------------------------------------------------------------------
    Returns
    out: plots
    """
    time_windows = [
    'win60_olap0',  
    'win30_olap0',
    'win20_olap0',
    'win22_olap10',
    'win20_olap10',# 5 per instance
    'win18_olap10',# 6 per instance
    'win15_olap8',
    ]  
    path = 'results/accuracies/' + condition + '/'
    condition_accuracies = load_condition_accuracies(path)
    cols, rows = 2, np.ceil(len(TimeParams_list)/2).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize = fig_sz, dpi = 100,
                            sharey = shareY)
    # Iter over conditions
    for TimeParam, ax in zip(TimeParams_list, fig.axes):
        # Iter over trees
        kernels = ['rbf', 'linear', 'sigmoid']
        curves = []
        for kernel_idx in range(3):
            accuracies = []
            lens_PerInst = [] 
            for win_olap_str in time_windows:
                accuracies_df = condition_accuracies[win_olap_str]['SVM']
                accuracies.append(accuracies_df.iloc[kernel_idx][TimeParam])
                lens_PerInst.append(get_len_PerInst(win_olap_str))
            curv = ax.plot(lens_PerInst, accuracies, marker = 'o')
            curves.append(curv)
        # Subplot text
        ax.set_title(TimeParms_ES[TimeParam], size = subplt_tit_sz )
        ax.set_xlabel('Datos por ciclo', size = subplt_XYlabel_sz)
        ax.set_ylabel('Accuracy', size = subplt_XYlabel_sz)
    fig.legend(curves, labels = kernels, ncol = 1, fontsize = 'large',
               loc = legend_loc)
    plt.tight_layout()
    plt.show()
#%%
def plot_SVM_Heatmap(condition, Kernel, TimeParams_list, win_olap_str,
                     Cparams_list, fig_sz = (12,10), tit_sz = 13,
                     subplt_tit_sz = 11, subplt_XYlabel_sz = 10,
                     cbar_orient = 'horizontal'):
    """
    Plotea las accuracies obtenidas variando los parámetro 'C' y 'gamma' para
    el SVC. Se plotea solo una cbar, ya que de lo contrario no es posible notar
    cambios en cada subplot
    --------------------------------------------------------------------------
    Parameters
    
    condition: str
        condition name.
    
    Kernel: str {'linear', 'rbf', 'sigmoid'}
        SVM kernel to be loaded a ploted.
    
    TimeParams_list: list
        List containing the names of the time parameter to be ploted.
        example: ['RMS', 'Mean']
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    
    Cparams_list: list of floats
        List of the C params to be ploted.
    
    fig_sz: tuple, default = ()
        Figure's size.
    
    tit_sz: float or int, default = 
        Superior title font size.
    
    subplt_tit_sz: float or int, deafult =
        Subplot title font title size.
    
    subplt_XYlabel_sz: float or int, deafult =
        Subplot xlabel and ylabel font size.
    
    cbar_orient: str {'horizontal', 'vertical'}
        Color bar orientation.
    --------------------------------------------------------------------------
    Returns
    out: plots
    """
    condiciones = {
        'cooler' : 'Estado del enfriador',
        'valve' : 'Estado de la válvula',
        'pump' : 'Fuga en la bomba',
        'accumulator' : 'Estado del acumulador',
        'stableFlag' : 'Estabilidad del sistema'
        }    

    TimeParms_ES ={
        'RMS' : 'RMS',
        'P2P' : 'Valor peak to peak',
        'Mean' : 'Media',
        'Variance' : 'Varianza'
        }
    #Load accuracies DataFrame
    path = 'results/accuracies/' + condition + '/'
    data = load_condition_accuracies(path)[win_olap_str]['SVM' + Kernel + 'Params']
    cols, rows = 2, np.ceil(len(TimeParams_list)/2).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize = fig_sz, dpi = 100)
    condicion = condiciones[condition]
    tit_upper = 'Accuracies obtenidas de la clasifación: {},'.format(condicion)
    n_PerInst = get_len_PerInst(win_olap_str)
    tit_lower = '\n usando {} dato(s) por ciclo'.format(n_PerInst)
    suptitle = tit_upper + tit_lower + 'y un kernel ' + Kernel
    fig.suptitle(suptitle, size = tit_sz)
    for TimeParam, ax in zip(TimeParams_list, fig.axes):
        TimeParam_df = data[['gamma', 'C', TimeParam]]
        TimeParam_df = TimeParam_df.loc[TimeParam_df['C'].isin(Cparams_list)]
        TimeParam_df = TimeParam_df.pivot('gamma', 'C', TimeParam)
        hm = sns.heatmap(TimeParam_df, ax = ax, annot= True, fmt = ".3f",
                         linewidths = 0.2,
                         cbar_kws={"orientation": cbar_orient}) 
        hm.set_title(TimeParms_ES[TimeParam], size = subplt_tit_sz )
#%%
def plot_SVM_Heatmap2(condition, Kernel, TimeParams_list, win_olap_str,
                      Cparams_list, fig_sz=(12,9), tit_sz = 12,
                      subplt_tit_sz = 11, subplt_XYlabel_sz = 10,
                      cbar_orient = 'horizontal'):
    """
    Plotea las accuracies obtenidas variando los parámetro 'C' y 'gamma' para
    el SVC. La idea es plotear 1 cbar para todos los subplots con el fin de
    comparar los desempeños de c/u.
    --------------------------------------------------------------------------
    Parameters
    
    condition: str
        condition name.
    
    Kernel: str {'linear', 'rbf', 'sigmoid'}
        SVM kernel to be loaded a ploted.
    
    TimeParams_list: list
        List containing the names of the time parameter to be ploted.
        example: ['RMS', 'Mean']
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.
    
    Cparams_list: list of floats
        List of the C params to be ploted.
    
    fig_sz: tuple, default = ()
        Figure's size.
    
    tit_sz: float or int, default = 
        Superior title font size.
    
    subplt_tit_sz: float or int, deafult =
        Subplot title font title size.
    
    subplt_XYlabel_sz: float or int, deafult =
        Subplot xlabel and ylabel font size.
    
    cbar_orient: str {'horizontal', 'vertical'}
        Color bar orientation.
    --------------------------------------------------------------------------
    Returns
    out: plots
    """
    condiciones = {
        'cooler' : 'Estado del enfriador',
        'valve' : 'Estado de la válvula',
        'pump' : 'Fuga en la bomba',
        'accumulator' : 'Estado del acumulador',
        'stableFlag' : 'Estabilidad del sistema'
        }    
    #Load accuracies DataFrame
    path = 'results/accuracies/' + condition + '/'
    data = load_condition_accuracies(path)[win_olap_str]['SVM' + Kernel + 'Params']
    # Figure settings
    cols, rows = 2, np.ceil(len(TimeParams_list)/2).astype(int)
    fig, axs = plt.subplots(rows, cols, figsize = fig_sz, dpi = 100)
    cbar_ax = fig.add_axes([0.2, .1, .6, .05])
    # Set plot suptitle
    condicion = condiciones[condition]
    tit_upper = 'Accuracies obtenidas de la clasifación: {},'.format(condicion)
    n_PerInst = get_len_PerInst(win_olap_str)
    tit_lower = '\n usando {} dato(s) por ciclo'.format(n_PerInst)
    suptitle = tit_upper + tit_lower + 'y kernel ' + Kernel
    fig.suptitle(suptitle, size = tit_sz)
    # Iter over time params
    for i, ax in zip(range(len((TimeParams_list))), fig.axes):
        TimeParam = TimeParams_list[i]
        TimeParam_df = data[['gamma', 'C', TimeParam]]
        TimeParam_df = TimeParam_df.loc[TimeParam_df['C'].isin(Cparams_list)]
        TimeParam_df = TimeParam_df.pivot('gamma', 'C', TimeParam)
        hm = sns.heatmap(TimeParam_df, ax = ax, annot= True, fmt = ".3f",
                         linewidths = 0.2,
                         cbar_kws={"orientation": cbar_orient},
                         cbar=i == 0, cbar_ax=None if i else cbar_ax) 
        hm.set_title(TimeParms_ES[TimeParam], size = subplt_tit_sz )
#%%
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