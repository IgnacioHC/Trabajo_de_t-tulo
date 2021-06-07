# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:06:04 2021

@author: Ignacio
"""
#%% Imports
import numpy as np
from utils.utils_preprocessing import get_Y
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
def get_XY_TrainTest(TimeParam_df, condition_labels):
    """
    Gets the time param data (X) and the labels (Y)  and splits it in train
    and test sets.
    --------------------------------------------------------------------------
    Parameters
        
    TimeParam_df: DataFrame
        DataFrame that contains the time param data

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance        
    --------------------------------------------------------------------------
    Returns
    out: tuple of 4 np.arrays
    X_train, X_test, Y_train, Y_test
    """ 
    X = TimeParam_df.values
    Y = get_Y(TimeParam_df, condition_labels=condition_labels)
    return train_test_split(X,Y, train_size=0.7,random_state=19)
#%%
def get_max_idxs(accuracies_list, max_accuracy):
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
            max_idxs.append(i)
    return max_idxs
#%%
def get_combinations(max_acc_idxs):
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
    for sensor_idx in sensors_idx_list:
        combinations_list.append(sorted(max_acc_idxs + [sensor_idx]))
    return combinations_list
#%%
def get_BestCombination(TimeParam_df, combinations_list, model, 
                        condition_labels):
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
    for sensors_combination in combinations_list:
        Model = model
        sensor_list = [all_sensors[i] for i in sensors_combination]
        X_train,X_test,Y_train,Y_test = get_XY_TrainTest(TimeParam_df[sensor_list],
                                                         condition_labels=condition_labels)
        Model.fit(X_train, Y_train)
        acc_list.append(accuracy_score(Y_test,Model.predict(X_test)))    
    max_acc = np.max(np.array(acc_list))
    #max_idxs = [sensors_idx_list[i] for i in get_max_idxs(acc_list,max_acc)]
    return max_acc
#%%
# def forward_selection(TimeParam_df, model, condition_labels,
#                       show_scores = False):
#     """
#     .
#     --------------------------------------------------------------------------
#     Parameters      
    
#     TimeParam_df: DataFrame
#         DataFrame that contains the time param data
    
#     model:
#         Model to make the calsification

#     condition_labels: np.array with shape (2205,)
#         Array that contains the class label for each instance     
#     --------------------------------------------------------------------------
#     Returns
#     out: 
#     """
#     good_idxs = []
#     old_max_acc = 0
#     max_acc = 0
#     while  max_acc >= old_max_acc:
#         old_max_acc = max_acc
#         combs_list = get_combinations(good_idxs)
#         max_idxs , max_acc = get_BestCombination(TimeParam_df=TimeParam_df,
#                                                  combinations_list=combs_list,
#                                                  model=model,
#                                                  condition_labels=condition_labels)
#         sensors_idx_list.remove(max_idxs[0])
#         good_idxs.append(max_idxs[0])
#     if show_scores == True:
#         print('Max accuracy: ',max_acc)
#         print('Mejores sensores: ',sorted(good_idxs))