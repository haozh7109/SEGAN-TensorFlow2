# -*- coding: utf-8 -*-
"""
Created on Mon. May 2 09:31:03 2022

@author: zhaoh
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil


# =============================================================================
# set the path, and split data to train, valid and test sets
# =============================================================================
# set parent Directory path
parent_dir = "/TensorFlowSSD/training_set_canteenFornebu/"

# set data directory
NoisyData_directory   = "noisy/"
CleanData_directory   = "clean/"

# set data paths
NoisyData_path   = os.path.join(parent_dir, NoisyData_directory)
CleanData_path   = os.path.join(parent_dir, CleanData_directory)

# get the filename list (sort both noisy and clean data by file-id number)
filenames_NoisyData = sorted(glob.glob(os.path.join(NoisyData_path, '*.wav')))
filenames_CleanData = sorted(glob.glob(os.path.join(CleanData_path, '*.wav')))

filenames_NoisyData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx,filename in enumerate(filenames_NoisyData)])
filenames_CleanData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx,filename in enumerate(filenames_CleanData)])

filenames_NoisyData_list = [filename for id, filename in filenames_NoisyData_sorted]
filenames_CleanData_list = [filename for id, filename in filenames_CleanData_sorted]

# split data list into train/validation/test li
train_NoisyData_list, test_NoisyData_list, train_CleanData_list, test_CleanData_list = train_test_split(filenames_NoisyData_list, filenames_CleanData_list, test_size=0.1, random_state=42)
train_NoisyData_list, valid_NoisyData_list, train_CleanData_list, valid_CleanData_list = train_test_split(train_NoisyData_list, train_CleanData_list, test_size=0.1, random_state=42)


# =============================================================================
# save data to different sub-folders
# =============================================================================

clean_train_folder = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/trainset_clean'
noisy_train_folder = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/trainset_noisy'
clean_test_folder  = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/testset_clean'
noisy_test_folder  = '/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/testset_noisy'

# save training and test data to different folders (clean and noisy)
# ps. for using the SEGAN code, the noisy and clean data will be set to the same file name with different sub-folders.
for clean, noisy in zip(train_CleanData_list, train_NoisyData_list):
    # get the unique file_id
    file_id_clean = clean.split("_")[-1].split(".wav")[0]
    file_id_noisy = noisy.split("_")[-1].split(".wav")[0]

    # copy the file to sub-folders
    if file_id_clean == file_id_noisy:
        print(f'--copying training used data : {clean}')
        shutil.copyfile(clean, clean_train_folder + f"/train_file_{file_id_clean}.wav")
        shutil.copyfile(noisy, noisy_train_folder + f"/train_file_{file_id_noisy}.wav")

    else:
        print(f'--could not find the file for : {clean}')


for clean, noisy in zip(test_CleanData_list, test_NoisyData_list):
    # get the unique file_id
    file_id_clean = clean.split("_")[-1].split(".wav")[0]
    file_id_noisy = noisy.split("_")[-1].split(".wav")[0]

    # copy the file to sub-folders
    if file_id_clean == file_id_noisy:
        print(f'--copying testing used data : {clean}')
        shutil.copyfile(clean, clean_test_folder + f"/test_file_{file_id_clean}.wav")
        shutil.copyfile(noisy, noisy_test_folder + f"/test_file_{file_id_noisy}.wav")

    else:
        print(f'--could not find the file for : {clean}')
