#%% Imports
import numpy as np
from itertools import combinations
import pickle
#%%
from itertools import combinations
get_comblist = lambda i:[list(comb) for comb in combinations(np.linspace(1,17,17).astype(int).tolist(),i)]
combinations_list = []
#%%
for i in range(1,18):
  combinations_list = combinations_list + get_comblist(i)
#%% save list
pickle.dump(combinations_list, open("data/combinations_list.txt", "wb"))