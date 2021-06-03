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
def get_XY_TrainTest(TimeParam_df,condition_labels):
  X = TimeParam_df.values
  Y = get_Y(TimeParam_df, condition_labels=condition_labels)
  return train_test_split(X,Y, train_size=0.7,random_state=19)
#%%
def get_combinations(sensors_idx_list,max_acc_idxs):
  combinations_list = [] 
  for sensor_idx in sensors_idx_list:
    combinations_list.append(sorted(max_acc_idxs + [sensor_idx]))
  return combinations_list
#%%
def get_max_idxs(accuracies_list,max_accuracy):
  max_idxs = []
  for i in range(len(accuracies_list)):
    if accuracies_list[i] == max_accuracy:
      max_idxs.append(i)
  return max_idxs
#%%
def get_BestCombination(TimeParam_df,combinations_list,model,condition_labels):
  acc_list = []
  for sensors_combination in combinations_list:
    Model = model
    sensor_list = [all_sensors[i] for i in sensors_combination]
    X_train,X_test,Y_train,Y_test = get_XY_TrainTest(TimeParam_df[sensor_list],
                                                    condition_labels=condition_labels)
    Model.fit(X_train,Y_train)
    acc_list.append(accuracy_score(Y_test,Model.predict(X_test)))
  max_acc = np.max(np.array(acc_list))
  max_idxs = [sensors_idx_list[i] for i in get_max_idxs(acc_list,max_acc)]
  return max_idxs , max_acc
#%%
def forward_selection(TimeParam_df,model,condition_labels):
  good_idxs = []
  old_max_acc = 0
  max_acc = 0
  while  max_acc >= old_max_acc:
    old_max_acc = max_acc
    combs_list = get_combinations(sensors_idx_list,good_idxs)
    max_idxs , max_acc = get_BestCombination(TimeParam_df=TimeParam_df,
                                             combinations_list=combs_list,
                                             model=model,
                                             condition_labels=condition_labels)
    sensors_idx_list.remove(max_idxs[0])
    good_idxs.append(max_idxs[0])
  print('Max accuracy: ',max)
  print('Mejores sesnores: ',sorted(good_idxs))