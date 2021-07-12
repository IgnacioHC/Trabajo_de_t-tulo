# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import numpy as np

from utils.utils_ModelsResults import load_condition_accuracies
from utils.utils_ModelsResults import plot_RF_accuracies
from utils.utils_ModelsResults import plot_KNN_accuracies
from utils.utils_ModelsResults import plot_SVM_accuracies
#%% DATA LOAD
condition_accuracies = load_condition_accuracies('results/accuracies/StableFlag/')
plot_SVM_accuracies(condition_accuracies,  ['RMS', 'P2P', 'Variance', 'Mean'])

#%%
