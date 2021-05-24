#%% Imports
import numpy as np
import pandas as pd
import pickle
#from itertools import combinations

from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#%% Load  data
df_RMS = pd.read_csv('data/data_TimeParams/data_RMS.csv')
combinations_list = pickle.load(open("data/combinations_list.txt", "rb"))
cooler_labels = np.loadtxt('data/labels/labels_cooler.txt')
#%%
keys = list(df_RMS.keys())
test_list = [combinations_list[i] for i in [15,16,17,18,19,20,21,22,23,24,25,26]]
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
#%%
#accuracies = get_scores(df_RMS,test_list,cooler_labels)