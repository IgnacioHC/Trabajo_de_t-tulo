#%% Imports
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from utils.utils_preprocessing import get_TimeParam_dict

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    pd.DataFrame.from_dict(datadict_TimeParam).to_csv(file_name,index=False)

#%%
get_comblist = lambda i:[list(comb) for comb in combinations(np.linspace(1,17,17).astype(int).tolist(),i)]
combinations_list = []
#%%
for i in range(1,18):
  combinations_list = combinations_list + get_comblist(i)
#%% save list
pickle.dump(combinations_list, open("data/combinations_list.txt", "wb"))
#%% Load  data
df_RMS = pd.read_csv('data/data_TimeParams/data_RMS.csv')
combinations_list = pickle.load(open("data/combinations_list.txt", "rb"))
cooler_labels = np.loadtxt('data/labels/labels_cooler.txt')
#%%
def get_Y_df(TimeParam_df,condition_labels):
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
keys = ['Temperature sensor 1','Temperature sensor 2','Temperature sensor 3',
'Temperature sensor 4','Vibration sensor','Cooling efficiency','Cooling power',
'Efficiency factor','Flow sensor 1','Flow sensor 2','Pressure sensor 1',
'Pressure sensor 2','Pressure sensor 3','Pressure sensor 4','Pressure sensor 5',
'Pressure sensor 6','Motor power']
def get_scores(TimeParam_df,combinations,condition_labels):
    acc_RF = []
    acc_KNN = []
    Y = get_Y_df(TimeParam_df, condition_labels)
    for combinations in combinations_list:
        comb_keys = [keys[i-1] for i in combinations]
        combination_df = TimeParam_df[comb_keys]
        X = combination_df.values
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                                                         train_size = 0.7,
                                                         random_state = 19)
        RF = RandomForestClassifier(n_estimators=100).fit(X_train,Y_train)
        KNN = KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
        acc_RF.append(accuracy_score(Y_test,RF.predict(X_test)))
        acc_KNN.append(accuracy_score(Y_test,KNN.predict(X_test)))
    return acc_RF , acc_KNN