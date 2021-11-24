# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:52:25 2021

@author: luu2
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from utils import add_operating_condition, add_remaining_useful_life, condition_scaler,\
    exponential_smoothing, gen_data_wrapper, gen_label_wrapper, gen_test_data_wrapper, scaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess(dataset,remaining_sensors,sequence_length,exp_smooth=0.1, style='LSTM', return_type='torch', seed = 42):

    if dataset == 'FD004' or dataset == 'FD002':
        multi_condition = True
    else:
        multi_condition = False

    dir_path = './CMAPSSData/'
    train_file = 'train_' + dataset + '.txt'
    test_file = 'test_' + dataset + '.txt'
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    col_names = index_names + setting_names + sensor_names
    train = pd.read_csv((dir_path+train_file), sep='\s+', header=None, 
                     names=col_names)
    test = pd.read_csv((dir_path+test_file), sep='\s+', header=None, 
                     names=col_names)
    y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep='\s+', header=None, 
                     names=['RemainingUsefulLife'])
    
    drop_sensors = [element for element in sensor_names if element not in remaining_sensors]
    
    # preprocessing data
    train = add_remaining_useful_life(train)
    
    
    train['RUL'].clip(upper=125, inplace=True)
    
    if multi_condition:
        X_train_interim = add_operating_condition(train.drop(drop_sensors, axis=1))
        X_test_interim = add_operating_condition(test.drop(drop_sensors, axis=1))
        
        X_train_interim, X_test_interim = condition_scaler(X_train_interim, X_test_interim, remaining_sensors)
    else:
        X_train_interim = train.drop(drop_sensors, axis=1)
        X_test_interim = test.drop(drop_sensors, axis=1)
        
        X_train_interim, X_test_interim = scaler(X_train_interim, X_test_interim, remaining_sensors)

    
    if exp_smooth>0:
        X_train_interim = exponential_smoothing(X_train_interim, remaining_sensors, 0, exp_smooth)
        X_test_interim = exponential_smoothing(X_test_interim, remaining_sensors, 0, exp_smooth)
    
    # train-val split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
    for train_unit, val_unit in gss.split(X_train_interim['unit_nr'].unique(), groups=X_train_interim['unit_nr'].unique()):
        train_unit = X_train_interim['unit_nr'].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_interim['unit_nr'].unique()[val_unit]
    
        train_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, train_unit)
        train_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], train_unit)
        
        val_split_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors, val_unit)
        val_split_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'], val_unit)
    
    # create sequences train, test 
    train_array = gen_data_wrapper(X_train_interim, sequence_length, remaining_sensors)
    train_label = gen_label_wrapper(X_train_interim, sequence_length, ['RUL'])
    
    test_array = gen_test_data_wrapper(df=X_test_interim, sequence_length=sequence_length, columns=remaining_sensors)
    
    # process train, test, validation labels
    train_label = train_label.squeeze()
    train_split_label = train_split_label.squeeze()
    val_split_label = val_split_label.squeeze()
    y_test = y_test.to_numpy()
    y_test = y_test.squeeze()
    y_test = y_test.astype('float32')
    
    
    if style == 'CNN':
        train_array = np.expand_dims(train_array,axis=3)
        test_array = np.expand_dims(test_array,axis=3)
        train_split_array = np.expand_dims(train_split_array,3)
        val_split_array = np.expand_dims(val_split_array,3)
    if style == 'FNN':
        n_ft = train_array.shape[2]*sequence_length
        train_array = np.reshape(train_array,(train_array.shape[0],n_ft))
        test_array = np.reshape(test_array,(test_array.shape[0],n_ft))
        train_split_array = np.reshape(train_split_array,(train_split_array.shape[0],n_ft))
        val_split_array = np.reshape(val_split_array,(val_split_array.shape[0],n_ft))

    if return_type == 'torch':
        train_array = torch.tensor(train_array, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_array = torch.tensor(test_array, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        train_split_array = torch.tensor(train_split_array, dtype=torch.float32)
        train_split_label = torch.tensor(train_split_label, dtype=torch.float32)
        val_split_array = torch.tensor(val_split_array, dtype=torch.float32)
        val_split_label = torch.tensor(val_split_label, dtype=torch.float32)

    return train_array, train_label, test_array, y_test, train_split_array, train_split_label, val_split_array, val_split_label
