# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:06:04 2021

@author: Ignacio
"""
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
#%%
