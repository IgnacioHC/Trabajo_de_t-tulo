# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:11:03 2021

@author: Ignacio
"""
#%% IMPORTS
import numpy as np
from numpy import mean, sqrt, square
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#%%
def to_TimeParam(Xt,time_param,t_win=20,overlap_ratio=0):
    """
    Calculates one of the time params (RMS , Variance, Mean) over time
    windows from sensor's instance data.   
    
    
    Parameters
    --------------------------------------------------------------------------
    
    Xt: 1D np.array
        Array that contains the time data of 1 instance measured by the sensor. 
        
    time_param:  {'RMS','Variance','Mean'}  
        Name of the time parameter to be calculated 
    
    t_win: int (t_win<60) ,default=20
        Window time length in seconds     
    
    overlap_ratio: float , default=0
        Ratio overlap/len_window
    
    Returns
    --------------------------------------------------------------------------
    out: numpy array with shape (win_per_instance,)
    
    """
    len_window = t_win/60 * Xt.shape[0] 
    overlap = len_window * overlap_ratio 
    win_per_instance= int(np.floor((Xt.shape[0]-overlap)/(len_window-overlap)))
        
    array_TimeParam = np.zeros((win_per_instance,))
    for i in range(win_per_instance):
        start = int(i*(len_window - overlap))
        stop = int(start + len_window)
        
        #Calculate the time param
        if time_param == 'RMS':
            array_TimeParam[i] = sqrt(mean(square(Xt[start:stop])))
        if time_param == 'P2P':
            array_TimeParam[i] = np.amax(Xt[start:stop])-np.amin(Xt[start:stop])    
        if time_param == 'Variance':
            array_TimeParam[i] = np.var(Xt[start:stop])      
        else: #time_param == 'Mean':
            array_TimeParam[i] = mean(Xt[start:stop])    
    return array_TimeParam
#%% ConditionsLabels_dict
"""
Create a dict that contains the name of the five  hydraulic system's
conditions as keys and another dict as values. Values dicts contains the name
of each class as keys and the labels of the classes as values. 
"""
ConditionsLabels_dict = {
    'Cooler condition':{
        'Close to total failure':3,
        'Reduced effifiency':20,
        'Full efficiency':100
    },
    'Valve condition':{
        'Optimal switching behavior':100,
        'Small lag':90,
        'Severe lag':80,
        'Close to total failure':73
    },
    'Pump leakage':{
        'No leakage':0,
        'Weak leakage':1,
        'Severe leakage':2
    },
    'Accumulator condition':{
        'Optimal pressure':130,
        'Slightly reduced pressure':115,
        'Severely reduced pressure':100,
        'Close to total failure':90
    },
    'Stable flag':{
        'Stable' : 0,
        'Not stable' : 1
        #'Conditions were stable':0,
        #'Static conditions might not have been reached yet':1
    }    
}
#%%
def get_TimeParam_dict(RawData_dict,time_param,t_win=20,overlap_ratio=0):
    """
    Calculates one of the time params over the time windows of each instance
    from the given sensor's raw data   
    
    
    Parameters
    --------------------------------------------------------------------------
    
    RawData_dict: dict
         Dict with the raw data from the sensors, in the form
         {sensor_name : data_raw}.  
        
    time_param:  {'RMS','Variance','Mean'}  
        Name of the time parameter to be calculated 
    
    t_win: int (t_win<60) , default=20
        Window time length in seconds     
    
    overlap_ratio: float , default=0
        Ratio overlap/len_window
    
    Returns
    --------------------------------------------------------------------------
    out: dict in the form {sensor_name : array_concat}. array_concat is the
    array that contains the time params with shape (win_per_instance*2205,).
    
    """
    TimeParam_dict = {}
    for sensor_name , sensor_data in RawData_dict.items():        
        array_concat = np.array([])
        for instance_idx in range(2205):
            array_TimeParam = to_TimeParam(sensor_data[instance_idx,:],
                                           time_param = time_param,
                                           t_win = t_win,
                                           overlap_ratio=overlap_ratio)            
            array_concat = np.concatenate((array_concat,array_TimeParam),axis=0)
        TimeParam_dict[sensor_name] = array_concat
    return TimeParam_dict
#%%
def split_classes(sensor_data,condition_name,condition_labels):   
    """
     Toma la data de un sensor como un 1D array, y la separa en sus
     respectivas clases

    Parameters
    --------------------------------------------------------------------------
    
    sensor_data: np.array
        Data del sensor con shape (2205*win_per_instance,)
        
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                     'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified       

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    Returns
    --------------------------------------------------------------------------
    out: dict 
      
    """    
    splited_classes = {}
    win_per_instance = int(sensor_data.shape[0]/2205)
    #Iterate over the condition's classes
    for class_name , class_label in ConditionsLabels_dict[condition_name].items():
        #Get a list with the indexes (from 0 to 2204) corresponding to the class
        class_oldindxs = np.where(condition_labels==class_label)[0].tolist()
        class_newindxs = np.array([])
        #Iterate over class indexes
        for old_idx in class_oldindxs:
            #Create the new indexes from 0 to 2205*win_per_instance-1
            new_idx = old_idx*win_per_instance + np.linspace(0,win_per_instance-1,win_per_instance)
            class_newindxs = np.concatenate((class_newindxs,new_idx),axis=0)
        splited_classes[class_name] = sensor_data[class_newindxs.astype(int)]
    return splited_classes
#%%
def plot_TimeParam(TimeParam_dict,condition_name,condition_labels,time_param
                   ,fig_sz=(20,18)):   
    """
    Plots the selected time parameter for each sensor data in data_dict as
    subplots in 1 figure. Every curve in each subplot represents a different
    class.

    Parameters
    --------------------------------------------------------------------------
    
    data_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam}.
       
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                      'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified    
        
    time_param:  {'RMS','Variance','Mean'}  
        Name of the time parameter to be calculated.    
        
    subplt: (n:int,m:int) tuple , optional
        Order of the subplots in the figure. total subplots = nxm.
        n=rows , m=cols
        
    fig_sz (float,float) tuple, default=(14,10)
        Size of the figure that contains the plots
        
        
    Returns
    --------------------------------------------------------------------------
    out: plots 
      
    """
    #Figure settings
    plt.figure(figsize=fig_sz , dpi=200)
    #Iterate over sensors
    for sensor_name in list(TimeParam_dict):
        #Subplot position
        i = list(TimeParam_dict).index(sensor_name) + 1
        plt.subplot(np.ceil(len(list(TimeParam_dict))/3).astype(int), 3,i)
        #Iterate over condition classes
        classes_dict = split_classes(sensor_data = TimeParam_dict[sensor_name],
                                     condition_name = condition_name,
                                     condition_labels = condition_labels)      
        for class_name , class_TimeParam_data in classes_dict.items():
            stop = class_TimeParam_data.shape[0]
            x = np.linspace(1, stop,stop)
            plt.scatter(x, class_TimeParam_data, label=class_name)
       #FigText
        title1 = time_param + ' obtenido del sensor '#+ TimeParam_df.columns[i]
        title2 = '\n Clasificasión: ' + condition_name 
        title = title1 + title2
        plt.title(title,size=10)
        plt.xlabel('Número de ventana temporal',size=8)
        plt.ylabel(time_param,size=8)    
        #Legend
        plt.legend()
    plt.tight_layout()
    plt.show()
#%%
def get_Y(TimeParam_df,condition_labels):
    """
     Toma el array con las etiquetas correspondientes a la clasificasión 
     (condición de salud) con shape (2205,5) y retorna las etiquetas con shape
     (2205*win_per_instance,)

    Parameters
    --------------------------------------------------------------------------

    Returns
    --------------------------------------------------------------------------
    out: np.array with shape (2205*win_per_instance,)

    """      
    #Calculate the number of windows per instance
    win_per_instance = int(len(TimeParam_df)/2205)
    Y_new = np.array([])
    #Iterate over the condition labels
    for label in condition_labels:
        #Create the new labels from 0 to 2205*win_per_instance-1
        new_labels = np.array([label]*win_per_instance)
        Y_new = np.concatenate((Y_new,new_labels),axis=0)
    return Y_new
#%%
def preprocess_data(RawData_dict,condition_labels,condition_name,time_param,
                    train_sz = 0.7,random_st=19):
    """
     Utiliza las funciones anteriormente definidas para realizar el
     preprocesamiento de los datos. Retorna los conjuntos X e Y

    Parameters
    --------------------------------------------------------------------------
    
    RawData_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam}.
       
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                     'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified   
        
    Returns
    --------------------------------------------------------------------------
    out: np.arrays
    Returns the X_train,X_test,Y_train,Y_test sets

    """
    #Get time param       
    TimeParam_dict = get_TimeParam_dict(RawData_dict=RawData_dict,
                                        time_param=time_param)
    #Scale data
    for sensor_name , sensor_data in TimeParam_dict.items():
      MinMaxScaler(copy=False).fit_transform(sensor_data.reshape(-1,1))
    #Get TimeParam_df
    TimeParam_df = pd.DataFrame.from_dict(TimeParam_dict)
    #Get X and Y
    X = TimeParam_df.values
    Y = get_Y(TimeParam_df, condition_labels=condition_labels)
    #Split data
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                                                     train_size = train_sz,
                                                     random_state = random_st)
    return X_train,X_test,Y_train,Y_test
#%%
