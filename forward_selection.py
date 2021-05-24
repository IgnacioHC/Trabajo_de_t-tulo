#%% Imports
import numpy as np
import pandas as pd
from itertools import combinations
#%% Load time params data
#df_RMS = pd.read_csv('/data/data_TimeParams/data_RMS.csv')
#%%
for i in range(17):
    combinations = combinations(np.linspace(1,17,17),i)
    for comb in combinations:
        print(comb)