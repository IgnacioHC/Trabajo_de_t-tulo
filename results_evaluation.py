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
from utils.utils_ModelsResults import plot_SVM_Heatmap
from utils.utils_ModelsResults import plot_SVM_Heatmap2
#%% DATA LOAD
condition = 'accumulator'
TimeParams_list = ['RMS', 'P2P', 'Variance', 'Mean']


win_olap_str = 'win60_olap0'
Kernel = 'linear'
Cs =  [1000, 10**4, 10**5, 10**6, 10**7]


conditions_list = [
    'cooler',
    'valve',
    'pump',
    'accumulator',
    'stableFlag'
    ]

# plot_SVM_Heatmap(condition, Kernel, TimeParams_list,
#                  win_olap_str,  Cparams_list = Cs)

#plot_SVM_accuracies(condition, TimeParams_list)
for condition in conditions_list:
    plot_RF_GiniEntro_accs(condition, TimeParams_list)