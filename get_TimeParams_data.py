"""
@author: Ignacio
"""
#%% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.utils_preprocessing import get_TimeParam_dict
#%% Load raw data
folder = 'data/data_raw/'
dataRaw_dict = {
    'Temperature sensor 1' : np.loadtxt(folder + "TS1.txt"),
    'Temperature sensor 2' : np.loadtxt(folder + "TS2.txt"),
    'Temperature sensor 3' : np.loadtxt(folder + "TS3.txt"),
    'Temperature sensor 4' : np.loadtxt(folder + "TS4.txt"),
    'Vibration sensor' : np.loadtxt(folder + "VS1.txt"),
    'Cooling efficiency' : np.loadtxt(folder + "CE.txt"),
    'Cooling power' : np.loadtxt(folder + "CP.txt"),
    'Efficiency factor' : np.loadtxt(folder + "SE.txt"),
    'Flow sensor 1' : np.loadtxt(folder + "FS1.txt"),
    'Flow sensor 2' : np.loadtxt(folder + "FS2.txt"),
    'Pressure sensor 1' : np.loadtxt(folder + "PS1.txt"),
    'Pressure sensor 2' : np.loadtxt(folder + "PS2.txt"),
    'Pressure sensor 3' : np.loadtxt(folder + "PS3.txt"),
    'Pressure sensor 4' : np.loadtxt(folder + "PS4.txt"),
    'Pressure sensor 5' : np.loadtxt(folder + "PS5.txt"),
    'Pressure sensor 6' : np.loadtxt(folder + "PS6.txt"),
    'Motor power' : np.loadtxt(folder + "EPS1.txt")
}
#%% Preprocess data
TimeParams_list = ['RMS','Variance','P2P','Mean']
for TimeParam in TimeParams_list:
    #Get time param
    datadict_TimeParam = get_TimeParam_dict(dataRaw_dict,TimeParam)
    #Scale data
    for sensor_name , sensor_data in datadict_TimeParam.items():
        MinMaxScaler(copy=False).fit_transform(sensor_data.reshape(-1,1))
    #To DataFrame and save data
    file_name = 'data/data_TimeParams/data_' + TimeParam + '.csv'
    pd.DataFrame.from_dict(datadict_TimeParam).to_csv(file_name)