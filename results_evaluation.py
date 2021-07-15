# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import numpy as np

from utils.utils_ModelsResults import load_condition_accuracies

from utils.utils_ModelsResults import plot_RF_accuracies
from utils.utils_ModelsResults import plot_RF_GiniEntro_accs

from utils.utils_ModelsResults import plot_KNN_accuracies
from utils.utils_ModelsResults import plot_KNN_UniDist_accs

from utils.utils_ModelsResults import plot_SVM_accuracies
#%% DATA LOAD
condition_accuracies = load_condition_accuracies('results/accuracies/accumulator/')
TimeParams_list = ['RMS', 'P2P', 'Variance', 'Mean']
time_windows = [
    'win60_olap0', # 1 per instance 
    'win30_olap0', # 2 per instance
    'win22_olap10',# 4 per instance
    'win15_olap8',# 7 per instance
    ]

plot_KNN_UniDist_accs(condition_accuracies,  TimeParams_list, time_windows,
                      fig_sz = (15,10))

#%%
