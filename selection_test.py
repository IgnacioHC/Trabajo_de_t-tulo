import numpy as np
import pandas as pd
import pickle
from forward_selection import get_scores
#%% Load  data
df_RMS = pd.read_csv('data/data_TimeParams/data_RMS.csv')
combinations_list = pickle.load(open("data/combinations_list.txt", "rb"))
cooler_labels = np.loadtxt('data/labels/labels_cooler.txt')
#%%
test_list = [combinations_list[i] for i in [15,16,17,18,19,20,21,22,23,24,25,26]]
accuracies = get_scores(df_RMS,test_list,cooler_labels)