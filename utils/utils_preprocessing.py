# -*- coding: utf-8 -*-
"""
Este archivo contiene funciones para obtener y plotear los parámetros de
tiempo, además de otras funciones necesarias para entregar los conjuntos
X_train, X_test, Y_train, Y_test.
"""
#%% IMPORTS
import numpy as np
from numpy import mean, sqrt, square
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils.utils_ModelsResults import get_len_PerInst
#%%
def to_TimeParam(Xt, time_param, t_win = 20, t_olap = 0):
    """
    Calculates one of the time params (RMS , Variance, Mean) over time
    windows from sensor's instance data.   
    --------------------------------------------------------------------------
    Parameters
    
    Xt: 1D np.array
        Array that contains the time data of 1 instance measured by the sensor. 
        
    time_param:  {'RMS', 'P2P', 'Variance', 'Mean'}  
        Name of the time parameter to be calculated.
    
    t_win: int (t_win<=60), default=20
        Window time length in seconds.     
    
    t_olap: float, default=0
        Overlap time length in seconds.
    
    Returns
    --------------------------------------------------------------------------
    out: np.array with shape (win_per_instance,)
    
    """
    len_window = t_win/60 * Xt.shape[0] 
    overlap = t_olap/60 * Xt.shape[0]  
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
def get_TimeParam_dict(RawData_dict, time_param, t_win = 20,
                       t_olap = 0):
    """
    Calculates one of the time params over the time windows of each instance
    from the given sensor's raw data.
    --------------------------------------------------------------------------
     Parameters
     
    RawData_dict: dict
         Dict with the raw data from the sensors, in the form
         {sensor_name : data_raw}.  
        
    time_param:  {'RMS', 'P2P', 'Variance', 'Mean'}  
        Name of the time parameter to be calculated 
    
    t_win: int (t_win<60) , default=20
        Window time length in seconds.
    
    t_olap: float , default=0
        Overlap time length in seconds.
    
    Returns
    --------------------------------------------------------------------------
    out: dictionary 
        Dict in the form {sensor_name : array_concat}. Array_concat is the
        array that contains the time params with shape (win_per_instance*2205,)  
    """
    TimeParam_dict = {}
    for sensor_name , sensor_data in RawData_dict.items():        
        array_concat = np.array([])     
        for instance_idx in range(sensor_data.shape[0]):
            array_TimeParam = to_TimeParam(sensor_data[instance_idx,:],
                                           time_param, t_win, t_olap)            
            array_concat = np.concatenate((array_concat,array_TimeParam), axis=0)
        TimeParam_dict[sensor_name] = array_concat
    return TimeParam_dict
#%%
def split_classes(sensor_data, condition_name, condition_labels):   
    """
     Toma la data de un sensor como un 1D array, y la separa en un array por
     cada clase, retornando los arrays de cada clase en un diccionario.
    --------------------------------------------------------------------------
    Parameters
    
    sensor_data: np.array
        Sensor data with shape (2205*win_per_instance,)
        
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                     'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified       

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    Returns
    --------------------------------------------------------------------------
    out: dictionary
        Dictionary in the form {class_name : class_data}
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
            new_idx = old_idx*win_per_instance + np.linspace(0, win_per_instance-1, 
                                                             win_per_instance)
            class_newindxs = np.concatenate((class_newindxs,new_idx),axis=0)
        splited_classes[class_name] = sensor_data[class_newindxs.astype(int)]
    return splited_classes
#%%
def plot_TimeParam(TimeParam_dict, condition_name, condition_labels,
                   time_param, fig_sz=(20,18)):   
    """
    Plots the selected time parameter for each sensor data in data_dict as
    subplots in 1 figure. Every curve in each subplot represents a different
    class.
    --------------------------------------------------------------------------
    Parameters
    
    TimeParam_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam_data}.
       
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                      'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified.

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    time_param:  {'RMS', 'P2P', 'Variance','Mean'}  
        Name of the time parameter to be calculated.    
        
    fig_sz (float,float) tuple, default=(20,18)
        Size of the figure that contains the plots.
    
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
        title1 = time_param + ' from ' + sensor_name
        title2 = '\n Classification: ' + condition_name 
        title = title1 + title2
        plt.title(title,size=10)
        plt.xlabel('Time window',size=8)
        plt.ylabel(time_param,size=8)    
        #Legend
        plt.legend()
    plt.tight_layout()
    plt.show()
#%%
def split_data(dataRaw_dict, condition_labels, train_sz=0.7, random_st=19):
    """
    --------------------------------------------------------------------------
    Parameters
    
    dataRaw_dict: dict
         Dict with the raw data from the sensors, in the form
         {sensor_name : data_raw}.

    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    train_sz: 0 <= float <= 1 , default=0.7
        Name of the time parameter to be calculated.    
        
    random_st: int, default=19
        random_state param for the train_test_split() function.
    
    Returns
    --------------------------------------------------------------------------
    out: tuple (RawData_dict_train, RawData_dict_test, Y_train, Y_test)
        Tuple with the train and test data as dictionaries and the train and
        test labels.
      
    """    
    RawData_dict_train, RawData_dict_test = {}, {} 
    for sensor_name, sensor_RawData in dataRaw_dict.items():
        sensor_sets = train_test_split(sensor_RawData, condition_labels,
                                       train_size = train_sz,
                                       random_state = random_st)
        RawData_sensor_train, RawData_sensor_test, Y_train, Y_test = sensor_sets
        RawData_dict_train[sensor_name] = RawData_sensor_train
        RawData_dict_test[sensor_name] = RawData_sensor_test
    return RawData_dict_train, RawData_dict_test, Y_train, Y_test
#%%
def get_Y(condition_labels, t_win = 20, t_olap = 0):
    """
     Toma el array con las etiquetas correspondientes a una condición a
     clasificar, con shape (2205,5) y retorna nuevas etiquetas con shape
     (2205*win_per_instance,)

    Parameters
    --------------------------------------------------------------------------
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
    
    t_win: int (t_win<60) , default=20
        Window time length in seconds.
    
    t_olap: float , default=0
        Overlap time length in seconds.

    Returns
    --------------------------------------------------------------------------
    out: np.array with shape (2205*win_per_instance,)

    """      
    #Calculate the number of windows per instance
    win_per_instance= int(np.floor((60-t_olap)/(t_win-t_olap)))
    Y_new = np.array([])
    #Iterate over the condition labels
    for label in condition_labels:
        #Create the new labels from 0 to 2205*win_per_instance-1
        new_labels = np.array([label]*win_per_instance)
        Y_new = np.concatenate((Y_new,new_labels),axis=0)
    return Y_new
#%%
def preprocess_data(RawData_dict, condition_labels, time_param, t_win, t_olap,
                    train_sz = 0.7, random_st=19):
    """
     Utiliza las funciones anteriormente definidas para realizar el
     preprocesamiento de los datos.

    Parameters
    --------------------------------------------------------------------------
    
    RawData_dict: dict
        Contains time param data for every sensor as
        {'sensor_name' : sensor_TimeParam}.
       
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance

    time_param:  {'RMS', 'P2P', 'Variance', 'Mean'}  
        Name of the time parameter to be calculated 
    
    t_win: int (t_win<60) , default=20
        Window time length in seconds.
    
    t_olap: float , default=0
        Overlap time length in seconds.

    train_sz: 0 <= float <= 1 , default=0.7
        Name of the time parameter to be calculated.    
        
    random_st: int, default=19
        random_state param for the train_test_split() function.
        
    Returns
    --------------------------------------------------------------------------
    out: np.arrays
        Returns the X_train,X_test,Y_train,Y_test sets

    """
    # Split data
    sets = split_data(RawData_dict, condition_labels, train_sz, random_st)
    RawData_dict_train, RawData_dict_test, Y_train, Y_test = sets
    #Get time param dicts
    TimeParam_dict_train = get_TimeParam_dict(RawData_dict_train, time_param,
                                              t_win, t_olap)
    TimeParam_dict_test = get_TimeParam_dict(RawData_dict_test, time_param,
                                             t_win, t_olap)
    #Scale data
    for sensor_name , sensor_train_data in TimeParam_dict_train.items():
        sensor_test_data = TimeParam_dict_test[sensor_name]
        #Concatenate train and test sensor data to fit the scaler
        concat_array = np.concatenate([sensor_train_data, sensor_test_data], axis=0)
        #Initialize and fit the scaler
        scaler_fit = MinMaxScaler(copy=False).fit(concat_array.reshape(-1,1))
        #Scale train and test data
        scaler_fit.transform(sensor_train_data.reshape(-1,1))
        scaler_fit.transform(TimeParam_dict_test[sensor_name].reshape(-1,1))
    #Get time param DataFrames
    TimeParam_df_train =  pd.DataFrame.from_dict(TimeParam_dict_train)
    TimeParam_df_test =  pd.DataFrame.from_dict(TimeParam_dict_test)
    #Get new labels with the shapes of the time windows
    Y_train = get_Y(Y_train, t_win, t_olap)
    Y_test = get_Y(Y_test, t_win, t_olap)
    return TimeParam_df_train, TimeParam_df_test, Y_train, Y_test
#%%
def plot_TimeParamES(TimeParam_dict, condition_name, condition_labels, 
                     time_param, win_olap_str, fig_sz=(10,9), tit_sz = 15,
                     dpi = 200, subplt_tit_sz = 12, subplt_XYlabel_sz = 12,
                     save_fig = False, tail_path2 ='.png'):   
    """
    Misma función que plot_TimeParam, pero plotea en español.
    """
    
    Parametros_tiempo = {
        'RMS':'RMS',
        'P2P':'P2P',
        'Variance':'Varianza',
        'Mean':'Media'}

    Nombres_clases = {
        'Cooler condition':{
            'Close to total failure': 'Cerca de la falla total',
            'Reduced effifiency':'Eficiencia reducida',
            'Full efficiency':'Eficiencia total'
        },
        'Valve condition':{
            'Optimal switching behavior': 'Comportamiento óptimo del switch',
            'Small lag': 'Pequeño retraso del switch',
            'Severe lag': 'Retraso severo del switch',
            'Close to total failure':'Cerca de la falla total'
        },
        'Pump leakage':{
            'No leakage': 'Sin fuga',
            'Weak leakage': 'Fuga leve',
            'Severe leakage': 'Fuga severa'
        },
        'Accumulator condition':{
            'Optimal pressure': 'Presión óptima',
            'Slightly reduced pressure': 'Presión ligeramente reducida',
            'Severely reduced pressure': 'Presión severamente reducida',
            'Close to total failure':'Cerca de la falla total'
        },
        'Stable flag':{
            'Stable' : 'Sistema estable',
            'Not stable' : 'Sistema no estable'
        }    
    }

    Nombre_sensores = {
          'Temperature sensor 1' : 'Sensor de temperatura 1',
          'Temperature sensor 2' : 'Sensor de temperatura 2',
          'Temperature sensor 3' : 'Sensor de temperatura 3',
          'Temperature sensor 4' : 'Sensor de temperatura 4',
          'Vibration sensor' : 'Sensor de vibración',
          'Cooling efficiency' : 'Eficiencia del enfriador',
          'Cooling power' : 'Potencia del enfriador',
          'Efficiency factor' : 'Factor de eficiencia',
          'Flow sensor 1' : 'Sensor de flujo de agua',
          'Flow sensor 2' : 'Sensor de flujo de agua',
          'Pressure sensor 1' : 'Sensor de presión 1',
          'Pressure sensor 2' : 'Sensor de presión 2',
          'Pressure sensor 3' : 'Sensor de presión 3',
          'Pressure sensor 4' : 'Sensor de presión 4',
          'Pressure sensor 5' : 'Sensor de presión 5',
          'Pressure sensor 6' : 'Sensor de presión 6',
          'Motor power' : 'Potencia del motor'
          }
    condiciones = {
        'Cooler condition' : 'Estado del enfriador',
        'Valve condition' : 'Estado de la válvula',
        'Pump leakage' : 'Fuga en la bomba',
        'Accumulator condition' : 'Estado del acumulador',
        'Stable flag' : 'Estabilidad del sistema'
        }    

    #Figure settings
    fig = plt.figure(figsize=fig_sz , dpi=dpi)
    n_PerInst = get_len_PerInst(win_olap_str)
    title_up = time_param  + ' para la clasificación: '
    title_low = '\nusando {} dato(s) por ciclo'.format(n_PerInst)
    suptitle = title_up + condiciones[condition_name] + ',' + title_low 
    fig.suptitle(suptitle+ '\n'+'\n'+'\n', size = tit_sz, ha = 'center')
    #Iterate over sensors
    for sensor_name in list(TimeParam_dict):
        #Subplot position
        i = list(TimeParam_dict).index(sensor_name) + 1
        rows = np.ceil(len(list(TimeParam_dict))/3).astype(int)
        plt.subplot(rows, 3,i)
        #Iterate over condition classes
        classes_dict = split_classes(sensor_data = TimeParam_dict[sensor_name],
                                     condition_name = condition_name,
                                     condition_labels = condition_labels)
        Labels = []
        for class_name , class_TimeParam_data in classes_dict.items():
            stop = class_TimeParam_data.shape[0]
            x = np.linspace(1, stop,stop)
            plt.scatter(x, class_TimeParam_data)
            Labels.append(Nombres_clases[condition_name][class_name])
        #FigText
        plt.title(Nombre_sensores[sensor_name], size = subplt_tit_sz)
        plt.xlabel('Número de ventana temporal', size = subplt_XYlabel_sz)
        plt.ylabel(Parametros_tiempo[time_param], size = subplt_XYlabel_sz)    
    #Legend
    if len(classes_dict) == 3:
        Ncol = 3
    else:
        Ncol = 2
    fig.legend(labels = Labels, bbox_to_anchor = (0.9, 0.9), ncol = Ncol,
               fancybox=True, fontsize = 'large')
    plt.tight_layout()
    if save_fig == True:
        head_path = 'images/TimeParams/' + time_param + '_'
        tail_path1 = condition_name.split(' ')[0] + '_' + win_olap_str
        plt.savefig(head_path + tail_path1 + tail_path2)
    else:
        pass
    plt.show()