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

from utils.utils_ModelsResults import plot_LDA_accuracies
#%% DATA LOAD

# condition = 'valve'
# TimeParams_list = ['RMS', 'P2P', 'Variance', 'Mean']
# plot_LDA_accuracies(condition, TimeParams_list)
