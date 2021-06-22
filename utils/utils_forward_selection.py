# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:06:04 2021

@author: Ignacio
"""
#%% Imports
import numpy as np
import pandas as pd
import json
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.utils_preprocessing import get_Y
from utils.utils_preprocessing import preprocess_data
#%% Sensor names list
all_sensors = ['Temperature sensor 1', #0
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
#%%
sensors_idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#%%
def get_max_idxs(accuracies_list, max_accuracy, sensors_idxs):
    """
    Returns the new index or indexes that gives the max accuracy in the curent
    forward selection iteration.
    --------------------------------------------------------------------------
    Parameters
        
    accuracies_list: list
        List of all the combination's accuracies from the current forward
        selection iteration.
            
    max_accuracy: float
        Maximun accuracy in the forward selection iteration.

    --------------------------------------------------------------------------
    Returns
    out: list of ints
    """ 
    max_idxs = []
    for i in range(len(accuracies_list)):
        if accuracies_list[i] == max_accuracy:
            max_idxs.append(sensors_idxs[i])
    return max_idxs
#%%
def get_combinations(max_acc_idxs, sensors_idxs):
    """
    Returns a list with the all the sensor's indexes that maximizes the
    accuracy in the current iteration of the forward selection.
    --------------------------------------------------------------------------
    Parameters      
    
    max_acc_idxs: list
        List with the index or indexes that gave the max accuracy in the last
        forward selection iteration.

    --------------------------------------------------------------------------
    Returns
    out: list of ints
    """ 
    combinations_list = [] 
    for sensor_idx in sensors_idxs:
        combinations_list.append(sorted(max_acc_idxs + [sensor_idx]))
    return combinations_list
#%%
def get_BestCombination(TimeParam_df_train, TimeParam_df_test, sensors_idxs,
                        good_idxs, model, Y_train, Y_test):
    """
    .
    --------------------------------------------------------------------------
    Parameters      
    
    TimeParam_df: DataFrame
        DataFrame that contains the time param data

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    acc_list = []
    for sensors_combination in get_combinations(good_idxs, sensors_idxs):
        Model = model
        sensor_list = [all_sensors[i] for i in sensors_combination]
        X_train = TimeParam_df_train[sensor_list].values
        X_test = TimeParam_df_test[sensor_list].values
        Model.fit(X_train, Y_train)
        acc_list.append(accuracy_score(Y_test, Model.predict(X_test)))    
    max_acc = np.max(np.array(acc_list))
    return max_acc , get_max_idxs(acc_list, max_acc, sensors_idxs)
#%%
def forward_selection(TimeParam_df, model, condition_labels,
                      show_scores = False):
    """
    .
    --------------------------------------------------------------------------
    Parameters      
    
    TimeParam_df: DataFrame
        DataFrame that contains the time param data
    
    model:
        Model to make the calsification

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance     
    --------------------------------------------------------------------------
    Returns
    out: 
    """
    good_idxs = []
    old_max_acc = 0
    max_acc = 0
    while  max_acc >= old_max_acc:
        old_max_acc = max_acc
        max_acc, max_idxs = get_BestCombination(TimeParam_df,
                                                sensors_idx_list,
                                                good_idxs,
                                                model,
                                                condition_labels)
        sensors_idx_list.remove(max_idxs[0])
        good_idxs.append(max_idxs[0])
    if show_scores == True:
        print('Max accuracy: ',max_acc)
        print('Mejores sensores: ',sorted(good_idxs))
    return old_max_acc
#%%
def forward_select(model, TimeParam_df_train, TimeParam_df_test, Y_train,
                   Y_test):
    start_time = time.time()
    sensors_idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    accuracies = []
    good_idxs = []
    #Iteration 1
    # if lda == True:
    #     Model = LinearDiscriminantAnalysis(n_components=1)
    #     max_acc, max_idxs = get_BestCombination(TimeParam_df, sensors_idx_list,
    #                                             good_idxs, Model,
    #                                             condition_labels)
    # else:
    #     pass
    max_acc, max_idxs = get_BestCombination(TimeParam_df_train, TimeParam_df_test,
                                            sensors_idx_list , good_idxs,
                                            model, Y_train, Y_test)
    accuracies.append(max_acc)
    
    # Iteration 2
    sensors_idx_list.remove(max_idxs[0])
    old_good_idxs = good_idxs
    good_idxs.append(max_idxs[0])
    # if lda == True and model != LinearDiscriminantAnalysis(n_components=1):
    #     Model = LinearDiscriminantAnalysis(n_components=2)
    #     max_acc, max_idxs = get_BestCombination(TimeParam_df, sensors_idx_list,
    #                                             good_idxs, Model,
    #                                             condition_labels)
    # else:
    #     pass
    max_acc, max_idxs = get_BestCombination(TimeParam_df_train, TimeParam_df_test,
                                            sensors_idx_list , good_idxs,
                                            model, Y_train, Y_test)
    accuracies.append(max_acc)
    
    # Iteration 3
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    # Iteration 4
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    # Iteration 5
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 6
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 7
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 8
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 9
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 10
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 11
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 12
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 13
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 14
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    
    #Iteration 15
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    #Iteration 16
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    #Iteration 17
    if accuracies[-1] >= accuracies[-2]:
        sensors_idx_list.remove(max_idxs[0])
        old_good_idxs = good_idxs
        good_idxs.append(max_idxs[0])
        max_acc, max_idxs = get_BestCombination(TimeParam_df_train, 
                                                TimeParam_df_test,
                                                sensors_idx_list , good_idxs,
                                                model, Y_train, Y_test)
        accuracies.append(max_acc)
    else:
        pass
    execution_time = time.time() - start_time
    return np.max(np.array(accuracies)), old_good_idxs, execution_time
#%%
def TimeParam_fwd_select(model, conditions_labels, TimeParams_df_dict):
    accuracies_dict = {}
    good_idxs_dict = {}
    times_dict = {}
    for TimeParam_name, TimeParam_data in TimeParams_df_dict.items():
        TimeParam_df_train = TimeParam_data['train_df']
        TimeParam_df_test = TimeParam_data['test_df']
        Y_train = TimeParam_data['Y_train']
        Y_test = TimeParam_data['Y_test']
        best_acc, good_idxs, exc_time = forward_select(model,
                                                       TimeParam_df_train,
                                                       TimeParam_df_test,
                                                       Y_train, Y_test)
        accuracies_dict[TimeParam_name] = best_acc
        good_idxs_dict[TimeParam_name] = good_idxs
        times_dict[TimeParam_name] = exc_time
    return accuracies_dict, good_idxs_dict, times_dict
#%%
def model_fwd_select(models_dict, conditions_labels, TimeParams_df_dict):
    model_series = []
    idx = []
    models_idxs_dict = {}
    times_series = []
    for model_name, model in models_dict.items():
        acc_dict, idxs_dict, times_dict = TimeParam_fwd_select(model,
                                                               conditions_labels,
                                                               TimeParams_df_dict)
        #Create a model's serie with the accuracies by time param
        model_series.append(pd.Series(acc_dict))
        times_series.append(pd.Series(times_dict))
        idx.append(model_name)
        #Add the max accuracy idxs dict to an another dict by model
        accuracies_DataFrame = pd.DataFrame(model_series, index = idx)
        models_idxs_dict[model_name] = idxs_dict
        times_DataFrame = pd.DataFrame(times_series, index = idx)
    return accuracies_DataFrame, models_idxs_dict, times_DataFrame
#%%
def save_accuracies(models_dict, condition_name, cond_accuracies,win_olap_str):
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head = 'results/accuracies/' + condition_name + '/' + win_olap_str + '/'
    tail = condition_name + '_' + models_name + '_' + win_olap_str + '.csv'
    file_path = head + tail
    cond_accuracies.to_csv(file_path)
#%%
def save_MaxAcc_idxs(condition_idxs, models_dict, win_olap_str):
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head_path = 'results/max_accuracy_idxs/' + win_olap_str + '/'
    tail_path = 'max_idxs_' + models_name + '_' + win_olap_str + '.json'
    file_path = head_path + tail_path
    with open(file_path, 'w') as fp:
        json.dump(condition_idxs, fp)
#%%
def save_times(models_dict, condition_name, cond_times, win_olap_str):
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head = 'results/execution_times/' + condition_name + '/' + win_olap_str
    tail = '/times_' + condition_name + '_' + models_name + '_' + win_olap_str + '.csv'
    file_path = head + tail
    cond_times.to_csv(file_path)    
#%%
def get_TimeParams_df_dict(TimeParams_list, RawData_dict, condition_labels,
                           t_win, t_olap, train_sz, random_st):
    TimeParams_df_dict = {}
    for time_param in TimeParams_list:
        sets = preprocess_data(RawData_dict, condition_labels, time_param,
                               t_win, t_olap, train_sz, random_st)
        TimeParam_df_train, TimeParam_df_test, Y_train, Y_test = sets
        TimeParams_df_dict[time_param] = {'train_df': TimeParam_df_train,
                                         'test_df': TimeParam_df_test,
                                         'Y_train': Y_train,
                                         'Y_test': Y_test}
    return TimeParams_df_dict
#%%
def conditions_fwd_select(RawData_dict, conditions_dict, models_dict,
                          TimeParams_list, win_olap_str, train_sz = 0.7,
                          random_st = 19, save = True):
    print('=====',str(win_olap_str),'=====')
    conditions_accuracies = {}
    condition_idxs = {}
    conditions_times = {}
    win = int(win_olap_str.split('_')[0][3:])
    olap = int(win_olap_str.split('_')[0][4:])
    for condition_name, condition_labels in conditions_dict.items():
        #Get TimeParams_df_dict
        TimeParams_df_dict = get_TimeParams_df_dict(TimeParams_list,
                                                    RawData_dict, 
                                                    condition_labels,
                                                    t_win = win,
                                                    t_olap = olap,
                                                    train_sz = train_sz,
                                                    random_st = random_st)
        #Forward selection
        accuracies, models_idxs, times = model_fwd_select(models_dict,
                                                          condition_labels,
                                                          TimeParams_df_dict)
        conditions_accuracies[condition_name] = accuracies
        condition_idxs[condition_name] = models_idxs
        conditions_times[condition_name] = times
        if save == True:
            save_accuracies(models_dict, condition_name, accuracies, win_olap_str)
            save_times(models_dict, condition_name, times, win_olap_str)
        else:
            pass
        print('FINISHED: ', condition_name)
    if save == True:
        save_MaxAcc_idxs(condition_idxs,models_dict, win_olap_str)
    return conditions_accuracies, condition_idxs, conditions_times