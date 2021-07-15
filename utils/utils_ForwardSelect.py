# -*- coding: utf-8 -*-
"""
"""
#%% Imports
import numpy as np
import pandas as pd
import json
import time

from sklearn.metrics import accuracy_score

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
    Returns the new index or indexes that gives the max accuracy in the
    current forward selection iteration.
    --------------------------------------------------------------------------
    Parameters
        
    accuracies_list: list
        List of all the combination's accuracies from the current forward
        selection iteration.
            
    max_accuracy: float
        Maximun accuracy in the forward selection iteration.
        
    sensors_idxs: list
        List with the indexes of all the sensors
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
    Returns a list with lists containing the all the sensor's indexes that
    maximizes the accuracy in the current iteration of the forward selection.
    --------------------------------------------------------------------------
    Parameters      
    
    max_acc_idxs: list
        List with the index or indexes that gave the max accuracy in the last
        forward selection iteration.
        
    sensors_idxs: list
        List with the indexes of all the sensors
    --------------------------------------------------------------------------
    Returns
    out: list of lists of ints
    """ 
    combinations_list = [] 
    for sensor_idx in sensors_idxs:
        combinations_list.append(sorted(max_acc_idxs + [sensor_idx]))
    return combinations_list
#%%
def get_BestCombination(TimeParam_df_train, TimeParam_df_test, sensors_idxs,
                        good_idxs, model, Y_train, Y_test):
    """
    Computa una iteración de forward selection. Para cada una de las
    combinaciones de sensores posibles, se entrena el modelo entregado y se
    calcula el accuracy.
    --------------------------------------------------------------------------
    Parameters      
    
    TimeParam_df: DataFrame
        DataFrame that contains the time param data

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance

    sensors_idxs: list
        List with the indexes of all the sensors
    
    good_idxs: list
        list of  the indexes that maximized the accuracy in the last forward
        selection iteration.
    
    model:
        model to be trained.
    
    Y_train: np.array
        train labels
    
    Y_test: np.array
        test labels
    --------------------------------------------------------------------------
    Returns
    out: tuple (max_acc, max_accuracy_indexes)
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
    DOES NOT WORK
    
    .
    --------------------------------------------------------------------------
    Parameters      
    
    TimeParam_df: DataFrame
        DataFrame that contains the time param data.
    
    model:
        Model to be trained.
        
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance. 
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
    """
    Using the previus functions, computes the forward selction for a certain
    model, time param and condition.
    --------------------------------------------------------------------------
    Parameters         
    
    model:
        Model to be trained.
    
    TimeParam_df_train: DataFrame
        DataFrame that contains the  time param train data.

    TimeParam_df_test: DataFrame
        DataFrame that contains the  time param test data.

    Y_train: np.array
        train labels
    
    Y_test: np.array
        test labels
    --------------------------------------------------------------------------
    Returns
    out: tuple (max_accuracy, max_accuracy_idxs, execution_time)
        Tuple with the maximun accuracy, the index or indexes corresponding to
        the max accuracy and the execution time for the forward selection.
    """
    start_time = time.time()
    sensors_idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    accuracies = []
    good_idxs = []
    max_acc, max_idxs = get_BestCombination(TimeParam_df_train, TimeParam_df_test,
                                            sensors_idx_list , good_idxs,
                                            model, Y_train, Y_test)
    accuracies.append(max_acc)
    
    # Iteration 2
    sensors_idx_list.remove(max_idxs[0])
    old_good_idxs = good_idxs
    good_idxs.append(max_idxs[0])
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
    """
    Using forward_select() function, computes the forward selction for all the
    given time params, for a certain model and condition.
    --------------------------------------------------------------------------
    Parameters         
    
    model:
        Model to be trained.

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance.
    
    TimeParam_df_dict: dictionary
        Dict that contains the time param data in the form:
        {TimeParam_name : {
            'X_train': TimeParam_df_train,
            'X_test': TimeParam_df_test,
            'Y_train' : Y_train,
            'Y_test' : Y_test}
            }
--------------------------------------------------
    Returns
    out: tuple (accuracies_dict, good_idxs_dict, times_dict)
    
        accuracies_dict: Dict in the form {TimeParam_name:max_accuracy}
        
        good_idxs_dict: Dict in the form {TimeParam_name:max_acc_idxs}
        
        times_dict: Dict in the form {TimeParam_name:execution_time}
    """
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
    """
    Using  TimeParam_fwd_select() function, computes the forward selction for
    all the given time params and all the given models for a certain condition.
    --------------------------------------------------------------------------
    Parameters         
    
    models_dict:
        Dict with the models to be trained.
        example: {model_name : model()}

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance.
    
    TimeParam_df_dict: dictionary
        Dict that contains the time param data in the form:
        {TimeParam_name : {
            'X_train': TimeParam_df_train,
            'X_test': TimeParam_df_test,
            'Y_train' : Y_train,
            'Y_test' : Y_test}
            }
--------------------------------------------------
    Returns
    out: tuple (accuracies_DataFrame, models_idxs_dict, times_DataFrame)
    
        accuracies_df: DF containing the accuracies, with the time params as
        columns and the models as rows.
    
        models_idxs_dict: Dict in the form {model_name : {
                                               {TimeParam_name:execution_time}
                                               }
        
        times_df: DF containing the execution times, with the time params as
        columns and the models as rows.
    """
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
def save_accuracies(models_dict, condition_name, cond_accuracies, win_olap_str):
    """
    Saves the accuracies obtained from the function conditions_fwd_select() as
    .csv files.
    --------------------------------------------------------------------------
    Parameters     

    models_dict: dictionary
        Dict with the models to be trained.
        example: {model_name : model()}
    
    condition_name: string {'cooler', 'valve', 'pump', 'accumulator',
                            'stableFlag'}
        Name of the condition to be clasified.
    
    cond_accuracies: DataFrame
        DF containing the accuracies, with the time params as columns and the
        models as rows.
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.

--------------------------------------------------
    Returns
    out: None
    """
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head = 'results/accuracies/' + condition_name + '/' + win_olap_str + '/'
    tail = condition_name + '_' + models_name + '_' + win_olap_str + '.csv'
    file_path = head + tail
    cond_accuracies.to_csv(file_path)
#%%
def save_MaxAcc_idxs(condition_idxs, models_dict, win_olap_str):
    """
    Saves the indexes that maximizes the accuracy by condition, model and time
    params as .json file.
    --------------------------------------------------------------------------
    Parameters
    
    condition_idxs: dictionary
        Dict containing the max accuracy indexes by model, condition and time
        param as {condition_name : {
                    model_name : {
                        TimeParam_name : execution_time
                        }
                    }
                }

    models_dict: dictionary
        Dict with the models to be trained.
        example: {model_name : model()}
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.

--------------------------------------------------
    Returns
    out: None
    """
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head_path = 'results/max_accuracy_idxs/' + win_olap_str + '/'
    tail_path = 'max_idxs_' + models_name + '_' + win_olap_str + '.json'
    file_path = head_path + tail_path
    with open(file_path, 'w') as fp:
        json.dump(condition_idxs, fp)
#%%
def save_times(models_dict, condition_name, cond_times, win_olap_str):
    """
    Saves the execution times obtained from the function
    conditions_fwd_select() as .csv files.
    --------------------------------------------------------------------------
    Parameters     

    models_dict:
        Dict with the models to be trained.
        example: {model_name : model()}
    
    condition_name: string {'cooler', 'valve', 'pump', 'accumulator',
                            'stableFlag'}
        Name of the condition to be clasified.
    
    cond_times: DataFrame
        DF containing the execution times, with the time params as columns and
        the models as rows.
    
    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.

--------------------------------------------------
    Returns
    out: None
    """
    models_name = list(models_dict.keys())[0].split(' ')[0]
    head = 'results/execution_times/' + condition_name + '/' + win_olap_str
    tail = '/times_' + condition_name + '_' + models_name + '_' + win_olap_str + '.csv'
    file_path = head + tail
    cond_times.to_csv(file_path)    
#%%
def get_TimeParams_df_dict(TimeParams_list, RawData_dict, condition_labels,
                           t_win, t_olap, train_sz, random_st):
    """
    Saves the execution times obtained from the function
    conditions_fwd_select() as .csv files.
    --------------------------------------------------------------------------
    Parameters
    
    TimeParams_list: list
        List containing the names of the time parameter to be calculated.
        example: ['RMS', 'Mean']
    
    RawData_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam}.
       
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
    
    t_win: int (t_win<60) , default=20
        Window time length in seconds.
    
    t_olap: float , default=0
        Overlap time length in seconds.

    train_sz: 0 <= float <= 1 , default=0.7
        Name of the time parameter to be calculated.    
        
    random_st: int, default=19
        random_state param for the train_test_split() function.
--------------------------------------------------
    Returns
    out: dictionary
    
    Dictionary in the form {TimeParam : 
                                {'train_df': TimeParam_df_train,
                                 'test_df': TimeParam_df_test,
                                 'Y_train': Y_train,
                                 'Y_test': Y_test
                                }
                            }
    """
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
    """
    Saves the execution times obtained from the function
    conditions_fwd_select() as .csv files.
    --------------------------------------------------------------------------
    Parameters
    
    RawData_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam}.
    
    conditions_dict:
        Dict with the conditions names and their condition labels to be
        trained.
        example: {condition_name : condition_labels}

    models_dict:
        Dict with the models to be trained.
        example: {model_name : model()}

    TimeParams_list: list
        List containing the names of the time parameter to be calculated.
        example: ['RMS', 'Mean']

    win_olap_str: string
        String with the window and overlap length in seconds
        example: 'win20_olap0', that means a window length of 20 seconds and
        an overlap length of 0 seconds.

    train_sz: 0 <= float <= 1 , default=0.7
        Name of the time parameter to be calculated.    
        
    random_st: int, default=19
        random_state param for the train_test_split() function.
    
    save: bool, default=True
        If True, uses the functions save_accuracies(), save_MaxAcc_idxs() and
        save_times() to save the results. If False does nothing.
--------------------------------------------------
    Returns
    out: dictionary
    
    Dictionary in the form {TimeParam : 
                                {'train_df': TimeParam_df_train,
                                 'test_df': TimeParam_df_test,
                                 'Y_train': Y_train,
                                 'Y_test': Y_test
                                }
                            }
    """
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
