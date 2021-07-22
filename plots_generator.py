# -*- coding: utf-8 -*-
"""
"""
#%% IMPORTS
import numpy as np

from utils.utils_ModelsResults import load_condition_accuracies
from utils.utils_ModelsResults import plot_RF_GiniEntro_accs
from utils.utils_ModelsResults import plot_SVM_accuracies
from utils.utils_ModelsResults import plot_SVM_Heatmap
from utils.utils_ModelsResults import plot_LDA_accuracies
#%% DATA LOAD
folder = 'data/data_raw/'
Raw_data = {
    'Temperature sensor 1' : np.loadtxt(folder + "TS1.txt"),
    'Temperature sensor 2' : np.loadtxt(folder + "TS2.txt"),
    'Temperature sensor 3' : np.loadtxt(folder + "TS3.txt"),
    'Temperature sensor 4' : np.loadtxt(folder + "TS4.txt"),
    'Vibration sensor' : np.loadtxt(folder + "VS1.txt"),
    'Cooling efficiency' : np.loadtxt(folder + "CE.txt"),
    'Cooling power' : np.loadtxt(folder + "CP.txt"),
    'Efficiency factor' : np.loadtxt(folder + "SE.txt"),
    'Flow sensor 1' : np.loadtxt(folder + "FS1.txt"),
    'Flow sensor 2' : np.loadtxt(folder + "FS2.txt"), #10
    'Pressure sensor 1' : np.loadtxt(folder + "PS1.txt"),
    'Pressure sensor 2' : np.loadtxt(folder + "PS2.txt"),
    'Pressure sensor 3' : np.loadtxt(folder + "PS3.txt"),
    'Pressure sensor 4' : np.loadtxt(folder + "PS4.txt"),
    'Pressure sensor 5' : np.loadtxt(folder + "PS5.txt"),
    'Pressure sensor 6' : np.loadtxt(folder + "PS6.txt"),
    'Motor power' : np.loadtxt(folder + "EPS1.txt")
}

#labels load
labels_folder = 'data/labels/'
conditions = {
    'cooler' : np.loadtxt(labels_folder + 'labels_cooler.txt'),
    'valve' : np.loadtxt(labels_folder + 'labels_valve.txt'),
    'pump' : np.loadtxt(labels_folder + 'labels_pump.txt'),
    'accumulator' : np.loadtxt(labels_folder + 'labels_accumulator.txt'),
    'stableFlag' : np.loadtxt(labels_folder + 'labels_stableFlag.txt')
    }


#%%
# RawData = {
#     'Temperature sensor 1' : Raw_data['Temperature sensor 1'],
#     'Temperature sensor 2' : Raw_data['Temperature sensor 2'],
#     'Temperature sensor 3' : Raw_data['Temperature sensor 3'],
#     'Temperature sensor 4' : Raw_data['Temperature sensor 4']
# }
# save = False
# from utils.utils_exploration import plt_RawSignalsES



# plt_RawSignalsES(RawData, 'Cooler condition', conditions['cooler'],
#                  subplt=(2,2), fig_sz = (10,5), save_fig=save,
#                  tail = 'Temperatures.png', bbox = (0.8, 0.9))

# plt_RawSignalsES(RawData, 'Valve condition', conditions['valve'],
#                   subplt=(2,2), fig_sz = (10,5), save_fig = save,
#                   tail = 'Temperatures.png', bbox = (0.8, 0.9))

# plt_RawSignalsES(RawData, 'Pump leakage', conditions['pump'], subplt=(2,2),
#                   fig_sz = (10,5), save_fig = save, tail = 'Temperatures.png')

# plt_RawSignalsES(RawData, 'Accumulator condition', conditions['accumulator'],
#                   subplt=(2,2), fig_sz = (10,5), save_fig = save,
#                   tail = 'Temperatures.png')

# plt_RawSignalsES(RawData, 'Stable flag', conditions['stableFlag'], subplt=(2,2),
#                   fig_sz = (10,5), save_fig = save, tail = 'Temperatures.png')
#%%
# RawData = {
#     'Flow sensor 1' : Raw_data['Flow sensor 1'],
#     'Flow sensor 2' : Raw_data['Flow sensor 2'],
# }
# from utils.utils_exploration import plt_RawSignalsES
# save = False


# plt_RawSignalsES(RawData, 'Cooler condition', conditions['cooler'],
#                  subplt=(1,2), fig_sz = (10,4.5), save_fig = save,
#                  tail = 'Flow.png', bbox = (0.8, 0.9))

# plt_RawSignalsES(RawData, 'Valve condition', conditions['valve'],
#                   subplt=(1,2), fig_sz = (10,4.5), save_fig = save,
#                   tail = 'Flow.png', bbox = (0.8, 0.9))

# plt_RawSignalsES(RawData, 'Pump leakage', conditions['pump'],
#                   subplt=(1,2), fig_sz = (10,4.5), save_fig = save,
#                   tail = 'Flow.png')

# plt_RawSignalsES(RawData, 'Accumulator condition', conditions['accumulator'],
#                   subplt=(1,2), fig_sz = (10,4.5), save_fig = save,
#                   tail = 'Flow.png')

# plt_RawSignalsES(RawData, 'Stable flag', conditions['stableFlag'],
#                   subplt=(1,2), fig_sz = (10,4.5), save_fig = save,
#                   tail = 'Flow.png')

#%%
# RawData = {
#     'Pressure sensor 1' : Raw_data['Pressure sensor 1'],
#     'Pressure sensor 2' : Raw_data['Pressure sensor 2'],
#     'Pressure sensor 3' : Raw_data['Pressure sensor 3'],
#     'Pressure sensor 4' : Raw_data['Pressure sensor 4'],
#     'Pressure sensor 5' : Raw_data['Pressure sensor 5'],
#     'Pressure sensor 6' : Raw_data['Pressure sensor 6'],
# }
# from utils.utils_exploration import plt_RawSignalsES
# save = False


# plt_RawSignalsES(RawData, 'Cooler condition', conditions['cooler'],
#                  subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                  tail = '_Pressure.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Valve condition', conditions['valve'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Pressure.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Pump leakage', conditions['pump'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Pressure.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Accumulator condition', conditions['accumulator'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Pressure.png', bbox = (0.8, 0.95))
#%%
# RawData = {
#     'Vibration sensor' :  Raw_data['Pressure sensor 1'],
#     'Cooling efficiency' : Raw_data['Cooling efficiency'],
#     'Cooling power' : Raw_data['Cooling power'],
#     'Efficiency factor' : Raw_data['Efficiency factor'],
#     'Motor power' : Raw_data['Motor power']
# }

# plt_RawSignalsES(RawData, 'Cooler condition', conditions['cooler'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Others.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Valve condition', conditions['valve'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Others.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Pump leakage', conditions['pump'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Others.png', bbox = (0.8, 0.95))

# plt_RawSignalsES(RawData, 'Accumulator condition', conditions['accumulator'],
#                   subplt=(3,2), fig_sz = (10,11), save_fig = save,
#                   tail = '_Others.png', bbox = (0.8, 0.95))

#%%
RawData = {
    'Temperature sensor 1' : Raw_data['Temperature sensor 1'],
    'Vibration sensor' :  Raw_data['Pressure sensor 1'],
    'Cooling efficiency' : Raw_data['Cooling efficiency'],
    'Cooling power' : Raw_data['Cooling power'],
    'Pressure sensor 1' : Raw_data['Pressure sensor 1'],
    'Motor power' : Raw_data['Motor power']
}
from utils.utils_preprocessing import get_TimeParam_dict
# win60olap0
win = 60
olap = 0
RMS_win60olap0 = get_TimeParam_dict(RawData,'RMS', t_win = win, t_olap = olap)
P2P_win60olap0 = get_TimeParam_dict(RawData,'P2P', t_win = win, t_olap = olap)
Var_win60olap0 = get_TimeParam_dict(RawData,'Variance', t_win = win, t_olap = olap)
Mean_win60olap0 = get_TimeParam_dict(RawData,'Mean', t_win = win, t_olap = olap)
# win30olap0
win = 30
olap = 0
RMS_win30olap0 = get_TimeParam_dict(RawData,'RMS', t_win = win, t_olap = olap)
P2P_win30olap0 = get_TimeParam_dict(RawData,'P2P', t_win = win, t_olap = olap)
Var_win30olap0 = get_TimeParam_dict(RawData,'Variance', t_win = win, t_olap = olap)
Mean_win30olap0 = get_TimeParam_dict(RawData,'Mean', t_win = win, t_olap = olap)
# win30olap0
win = 18
olap = 10
RMS_win18olap10 = get_TimeParam_dict(RawData,'RMS', t_win = win, t_olap = olap)
P2P_win18olap10 = get_TimeParam_dict(RawData,'P2P', t_win = win, t_olap = olap)
Var_win18olap10 = get_TimeParam_dict(RawData,'Variance', t_win = win, t_olap = olap)
Mean_win18olap10 = get_TimeParam_dict(RawData,'Mean', t_win = win, t_olap = olap)
#%%
#===============COMPARACION ENTRE TIME PARAMS===============================
# save = True
# from utils.utils_preprocessing import plot_TimeParamES




# # COOLER
# win_olap_str = 'win60_olap0'
# plot_TimeParamES(RMS_win60olap0, 'Cooler condition', conditions['cooler'],
#                  'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(P2P_win60olap0, 'Cooler condition', conditions['cooler'],
#                  'P2P', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(Mean_win60olap0, 'Cooler condition', conditions['cooler'],
#                   'Mean', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(Var_win60olap0, 'Cooler condition', conditions['cooler'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# VALVE
# win_olap_str = 'win60_olap0'
# plot_TimeParamES(RMS_win60olap0, 'Valve condition', conditions['valve'],
#                  'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(P2P_win60olap0, 'Valve condition', conditions['valve'],
#                   'P2P', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(Mean_win60olap0, 'Valve condition', conditions['valve'],
#                   'Mean', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# plot_TimeParamES(Var_win60olap0, 'Valve condition', conditions['valve'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))



# ==========COMPARACION ENTRE CLASIFICACIONES=========================
# win_olap_str = 'win60_olap0'
# # PUMP
# plot_TimeParamES(RMS_win60olap0, 'Pump leakage', conditions['pump'],
#                   'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))


# plot_TimeParamES(Var_win60olap0, 'Pump leakage', conditions['pump'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))


# #ACCUMULATOR
# plot_TimeParamES(RMS_win60olap0, 'Accumulator condition', conditions['accumulator'],
#                   'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))


# plot_TimeParamES(Var_win60olap0, 'Accumulator condition', conditions['accumulator'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# =================COMPARACION ENTRE VENTANAS=======================
# COOLER

# win_olap_str = 'win30_olap0'
# plot_TimeParamES(RMS_win30olap0, 'Cooler condition', conditions['cooler'],
#                   'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# win_olap_str = 'win18_olap10'
# plot_TimeParamES(RMS_win18olap10, 'Cooler condition', conditions['cooler'],
#                   'RMS', win_olap_str, save_fig = save, fig_sz = (10, 6.5))
# #VALVE

# win_olap_str = 'win30_olap0'
# plot_TimeParamES(Var_win30olap0, 'Valve condition', conditions['valve'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

# win_olap_str = 'win18_olap10'
# plot_TimeParamES(Var_win18olap10, 'Valve condition', conditions['valve'],
#                   'Variance', win_olap_str, save_fig = save, fig_sz = (10, 6.5))

#%%
#============================== LDA =======================================
from utils.utils_ModelsResults import plot_LDA_accuracies
TimeParams_list = ['RMS', 'P2P', 'Variance', 'Mean']
save = True


plot_LDA_accuracies('cooler', TimeParams_list, fig_sz = (10 , 6),
                    save_fig = save)
plot_LDA_accuracies('valve', TimeParams_list, fig_sz = (10 , 6),
                    save_fig = save)
plot_LDA_accuracies('pump', TimeParams_list, fig_sz = (10 , 6),
                    save_fig = save)
plot_LDA_accuracies('accumulator', TimeParams_list, fig_sz = (10 , 6),
                    save_fig = save)
#%%
# ================================ KNN =======================================
# from utils.utils_ModelsResults import plot_RF_GiniEntro_accs
# TimeParams_list = ['RMS', 'P2P', 'Variance']
# save = True


# plot_RF_GiniEntro_accs('cooler', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                         save_fig = save)

# plot_RF_GiniEntro_accs('valve', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                         save_fig = save)

# plot_RF_GiniEntro_accs('pump', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                         save_fig = save)

# plot_RF_GiniEntro_accs('accumulator', TimeParams_list, n_cols = 3,
#                        fig_sz = (10,7), save_fig = save)

# TimeParams_list = ['RMS', 'P2P']

# plot_RF_GiniEntro_accs('accumulator', TimeParams_list, n_cols = 2,
#                        fig_sz = (10,5.5), save_fig = save)
# plot_RF_GiniEntro_accs('pump', TimeParams_list, n_cols = 2, fig_sz = (10,5.5),
#                        save_fig = save)
#%%
# ================================ RF =======================================
# from utils.utils_ModelsResults import plot_KNN_UniDist_accs
# TimeParams_list = ['RMS', 'P2P', 'Variance']
# save = True


# plot_KNN_UniDist_accs('cooler', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                       save_fig = save)
# plot_KNN_UniDist_accs('valve', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                       save_fig = save)
# plot_KNN_UniDist_accs('pump', TimeParams_list, n_cols = 3, fig_sz = (10,8),
#                       save_fig = save)
# plot_KNN_UniDist_accs('accumulator', TimeParams_list, n_cols = 3,
#                       fig_sz = (10,8), save_fig = save)

# TimeParams_list = ['RMS', 'P2P']
# plot_KNN_UniDist_accs('pump', TimeParams_list, n_cols = 2, fig_sz = (10,5.5),
#                       save_fig = save)
# plot_KNN_UniDist_accs('accumulator', TimeParams_list, n_cols = 2,
#                       fig_sz = (10,5.5), save_fig = save)

#%%
# ================================ SVM =======================================

# from utils.utils_ModelsResults import plot_SVM_accuracies
# from utils.utils_ModelsResults import plot_SVM_Heatmap
# TimeParams_list = ['RMS', 'Variance']
# save = False

# # acc vs win_len
# height = 3.5
# plot_SVM_accuracies('cooler', TimeParams_list, fig_sz = (10, height))
# plot_SVM_accuracies('valve', TimeParams_list, fig_sz = (10, height))
# plot_SVM_accuracies('pump', TimeParams_list, fig_sz = (10, height))
# plot_SVM_accuracies('accumulator', TimeParams_list, fig_sz = (10, height))
#%%
# # heatmaps
# TimeParams_list = ['RMS', 'Variance']
# Kernel = 'rbf'
# C_params = [1000, 10**4, 10**5, 10**6]
# win_olap_str = 'win60_olap0'


# plot_SVM_Heatmap('cooler', Kernel, TimeParams_list, win_olap_str, C_params,
#                  fig_sz = (10, 7))
# plot_SVM_Heatmap('valve', Kernel, TimeParams_list, win_olap_str, C_params,
#                  fig_sz = (10, 7))
# plot_SVM_Heatmap('pump', Kernel, TimeParams_list, win_olap_str, C_params,
#                  fig_sz = (10, 7))
# plot_SVM_Heatmap('accumulator', Kernel, TimeParams_list, win_olap_str, C_params,
#                  fig_sz = (10, 7))

#%%