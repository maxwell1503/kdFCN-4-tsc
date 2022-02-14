import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import PATH_DATA

def read_all_datasets(split_val=False):
    datasets_dict = {}
    cur_root_dir = PATH_DATA
    
    
    for dataset_name in DATASET_NAMES_2018:
        root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'
    
        df_train = pd.read_csv(root_dir_dataset + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
        df_test = pd.read_csv(root_dir_dataset + dataset_name + '_TEST.tsv', sep='\t', header=None)   

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]
        
        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])
        
        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])
        
        x_train = x_train.values
        x_test = x_test.values
        
        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
        
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    
    
    return datasets_dict

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_dataset(root_dir, dataset_name):
    datasets_dict = {}
    cur_root_dir = PATH_DATA
    root_dir_dataset = cur_root_dir + '/' + dataset_name + '/'

    df_train = pd.read_csv(root_dir_dataset + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    df_test = pd.read_csv(root_dir_dataset + dataset_name + '_TEST.tsv', sep='\t', header=None)
    
    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]
    
    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])
    
    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])
    
    x_train = x_train.values
    x_test = x_test.values
    
    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_
    
    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())
    
    return datasets_dict
