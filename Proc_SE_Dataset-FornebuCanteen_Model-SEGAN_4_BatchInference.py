# -*- coding: utf-8 -*-
"""
Created on Friday, March 11 10:36:03 2022
@author: zhaoh
"""

import glob
import os
import time
import multiprocessing

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy

import pandas as pd
from numpy import savetxt
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Py_data_Preprocess_Hao import SpectrogramProcess, WaveformDataGenerator, emphasis, emphasis_librosa
from Py_data_loader_Hao import DataLoader
from Test_Customized_CRN_model import GeneratorSegan, DiscriminatorSegan, VirtualBatchNorm
from tqdm import tqdm

###############################################################################
#  (0) run GPU version of model on CPU only to utilize the multi-cpus
###############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###############################################################################
#  (1) set parameters for spectrogram generation and data loading
###############################################################################
n_fft                = 128
hop_length           = 32
sample_rate          = 16000
segment_len          = 8
train_epochs         = 20
training_flag        = False
inference_batch_size = 100
window_size          = 2 ** 14  # about 1 second of samples
stride               = 0.5
QC_PLOT              = False

###############################################################################
#  (2) get the dataset for training
###############################################################################
# set parent Directory path
parent_dir = "/TensorFlowSSD/training_set_canteenFornebu/"

# set data directory
NoisyData_directory   = "noisy/"
CleanData_directory   = "clean/"
DenoiseData_directory = "denoised/"
QC_directory          = "QC/"

# set data paths
NoisyData_path   = os.path.join(parent_dir, NoisyData_directory)
CleanData_path   = os.path.join(parent_dir, CleanData_directory)
DenoiseData_path = os.path.join(parent_dir, DenoiseData_directory)
QC_path          = os.path.join(parent_dir, QC_directory)
MODEL_WEIGHTS_SAVE_PATH = os.path.join("/home/hao_listen/PycharmProjects/pythonProject/Model_Current_Training/SEGAN/model_weights_saved")

# get the filename list
filenames_NoisyData = sorted(glob.glob(os.path.join(NoisyData_path, '*.wav')))
filenames_CleanData = sorted(glob.glob(os.path.join(CleanData_path, '*.wav')))

filenames_NoisyData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx, filename in enumerate(filenames_NoisyData)])
filenames_CleanData_sorted = sorted([(filename.split("_")[-1].split(".wav")[0], filename) for indx, filename in enumerate(filenames_CleanData)])

filenames_NoisyData_list = [filename for id, filename in filenames_NoisyData_sorted]
filenames_CleanData_list = [filename for id, filename in filenames_CleanData_sorted]

# split data list into train/validation/test list
train_NoisyData_list, test_NoisyData_list, train_CleanData_list, test_CleanData_list = train_test_split(filenames_NoisyData_list, filenames_CleanData_list, test_size=0.1, random_state=42)

###############################################################################
#  (3) Generate the QC files
###############################################################################
# # load the previous trained model for QC generation
QC_selected_models     = ['SE_Dataset-FornebuCanteen_Model-Baseline_200epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline2Frames_200epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline_AugmentTime_100epochs', 'SE_Dataset-FornebuCanteen_Model-Baseline_AugmentTimeSpec_200epochs', 'SE_Dataset-FornebuCanteen_Model-Test_Architect_200epochs','SEGAN']
QC_selected_model      = QC_selected_models[-1]

if QC_selected_model == QC_selected_models[0]:
    model_name = 'Baseline'
elif QC_selected_model == QC_selected_models[1]:
    model_name = 'Baseline2Frames'
    segment_len = 2
elif QC_selected_model == QC_selected_models[2]:
    model_name = 'Baseline_AugmentTime'
elif QC_selected_model == QC_selected_models[3]:
    model_name = 'Baseline_AugmentTimeSpec'
elif QC_selected_model == QC_selected_models[4]:
    model_name = 'Test_Architect'
else:
    model_name = 'SEGAN'

# define the denoising-inference method, which will be used for multiple-cpu execution
def denoise_model_inference_func(job_id):

    # import tensorflow inside the function which requires by the multiple-cpu setting in python
    import tensorflow as tf

    # Get the generator and the corresponding trained model
    print('Defining the generator model...')
    generator = GeneratorSegan()

    print('get the latest weights...')
    for epoch in range(train_epochs, 0, -1):
        if os.path.exists(f'{MODEL_WEIGHTS_SAVE_PATH}/generator_weights_epoch_{epoch}.index'):
            generator.load_weights(f'{MODEL_WEIGHTS_SAVE_PATH}/generator_weights_epoch_{epoch}')
            print('load weights from epoch %d' % epoch)
            break

    # select qc noisy file
    QC_NoisyFile_lists = test_NoisyData_list[:inference_batch_size]
    noisy_file         = QC_NoisyFile_lists[job_id]

    # generate the QC file
    if not os.path.exists(os.path.join(QC_path + 'Lock/', 'lock_process_snr_fileID_' + str(job_id) + '_.lck')):

        # generate the lock key
        with open(os.path.join(QC_path + 'Lock/', 'lock_process_snr_fileID_' + str(job_id) + '_.lck'), 'w') as fp:
            pass

        #load noisy data
        noisy_signal, sample_rate_NoisyData = librosa.load(noisy_file, sr=sample_rate)
        clean_signal, sample_rate_CleanData = librosa.load(CleanData_path + '/clean_fileid' + noisy_file.split('/')[-1].split('fileid')[-1], sr=sample_rate)

        # slice the data into small pieces
        Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)
        noisy_slices = Waveform_generator.slice_signal(noisy_signal, window_size=window_size, stride=stride, sample_rate=sample_rate, waveform_type='loaded', padding=True)
        clean_slices = Waveform_generator.slice_signal(clean_signal, window_size=window_size, stride=stride, sample_rate=sample_rate, waveform_type='loaded', padding=True)

        # denoise the data
        denoised_slices = []
        clean_slices_qc = []
        noisy_slices_qc = []

        for noisy_slice, clean_slice in tqdm(zip(noisy_slices, clean_slices), desc='Apply audio enhancement to noisy data'):

            # get the preconditioned noisy data
            noisy_slice = emphasis_librosa(noisy_slice[np.newaxis, np.newaxis, :], emph_coeff=0.95)
            clean_slice = emphasis_librosa(clean_slice[np.newaxis, np.newaxis, :], emph_coeff=0.95)

            noisy_slice = tf.transpose(np.array(noisy_slice), [0, 2, 1])
            clean_slice = tf.transpose(np.array(clean_slice), [0, 2, 1])

            # generate random noise
            z = tf.random.normal(shape=(1, 8, 1024))

            # get the enhanced data
            denoised_slice = generator(noisy_slice, z)

            # remove the emphasis after the denoising
            denoised_slice = tf.transpose(np.array(denoised_slice), [0, 2, 1])
            clean_slice = tf.transpose(np.array(clean_slice), [0, 2, 1])
            noisy_slice = tf.transpose(np.array(noisy_slice), [0, 2, 1])

            # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # # beaware of the old implemenation has a bug with the emphasis and de-emphasis which are not reversible, thus the preempasis is not removed after the deemphasis. the amplitude of the output is much weaker than the original input!!!!
            # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # denoised_slice = emphasis(denoised_slice, emph_coeff=0.95,pre=False)
            # clean_slice    = emphasis(clean_slice, emph_coeff=0.95,pre=False)
            # noisy_slice    = emphasis(noisy_slice, emph_coeff=0.95,pre=False)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # use the following code to remove the emphasis after the deemphasis, implemenation from librosa
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            denoised_slice = emphasis_librosa(denoised_slice.numpy(), emph_coeff=0.95, pre=False)
            clean_slice = emphasis_librosa(clean_slice.numpy(), emph_coeff=0.95, pre=False)
            noisy_slice = emphasis_librosa(noisy_slice.numpy(), emph_coeff=0.95, pre=False)

            # output the denoised data
            denoised_slices.append(denoised_slice)
            noisy_slices_qc.append(noisy_slice)
            clean_slices_qc.append(clean_slice)

            # QC on a single data batch
            if QC_PLOT:
                plt.figure()
                plt.plot(np.reshape(noisy_slice, -1), label='noisy')
                plt.plot(np.reshape(clean_slice, -1), label='clean')
                plt.plot(np.reshape(denoised_slice, -1), label='denoised')
                plt.grid()
                plt.legend()
                plt.title('QC on a single denoised slice')
                plt.show()


        # save the denoised data
        Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)
        denoised_sound     = Waveform_generator.reconstruct_from_slice(denoised_slices, window_size=window_size, stride=stride, orginal_data_length=len(noisy_signal))
        clean_sound_qc     = Waveform_generator.reconstruct_from_slice(clean_slices_qc, window_size=window_size, stride=stride, orginal_data_length=len(noisy_signal))
        noisy_sound_qc     = Waveform_generator.reconstruct_from_slice(noisy_slices_qc, window_size=window_size, stride=stride, orginal_data_length=len(noisy_signal))

        if QC_PLOT:
            # QC on a single data batch
            plt.figure()
            plt.plot(noisy_sound_qc, label='noisy')
            plt.plot(denoised_sound, label='denoised')
            plt.plot(clean_sound_qc, label='clean')
            plt.grid()
            plt.legend()
            plt.title('QC on a single denoised sample')
            plt.show()

        # writout the denoised data
        print('writing denoised data to {}'.format(QC_path))
        #  save the output as wav files
        sf.write(QC_path + 'Clean_' + noisy_file.split('/')[-1], clean_sound_qc, sample_rate)
        sf.write(QC_path + 'Noisy_' + noisy_file.split('/')[-1], noisy_sound_qc, sample_rate)
        sf.write(QC_path + 'QC_Model-' + model_name + '_NN_Denoised_' + noisy_file.split('/')[-1], denoised_sound, sample_rate)

    else:
        print("-- Denoising process has applied already --")

###############################################################################
#  (4) Generate the QC files
###############################################################################

if __name__ ==  '__main__':

    print('-----------------------------------------------------------')
    print('----------Start denoising : ----------')
    print('-----------------------------------------------------------')

    start = time.time()

    # ## run single inference for test
    # denoise_model_inference_func(10)

    ## set up the multiprocess based job-execution
    # (1) option-1 : use process
    processes = []

    for job_id in np.arange(0, inference_batch_size):
        P = multiprocessing.Process(target=denoise_model_inference_func, args=(job_id,))
        processes.append(P)
        P.start()

    for process in processes:
        process.join()

    # # (2) option-2 : use pool
    # pool = Pool()
    # pool.map(denoise_model_inference_func, range(0, inference_batch_size))
    # pool.close()

    running_time = time.time() - start
    print('-----------------------------------------------------------')
    print("Denoising completed in : {:0.2f}s".format(running_time))
    print('-----------------------------------------------------------')
