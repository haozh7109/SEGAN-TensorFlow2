# -*- coding: utf-8 -*-
"""
Created on Mon. Jan 4 12:54:03 2022

@author: zhaoh
"""

import tensorflow as tf
import os
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import librosa, librosa.display
import soundfile as sf
from Utility.Py_data_Preprocess_Hao import SpectrogramProcess, WaveformDataGenerator, emphasis, emphasis_librosa
from Utility.model import GeneratorSegan, DiscriminatorSegan, VirtualBatchNorm
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Audio Enhancement with SEGAN')
    parser.add_argument('--batch_size', default=100, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
    parser.add_argument('--test_clean_data_path', default='/SSD/training_set_canteenFB/Data_for_SEGAN/testset_clean', type=str, help='test clean data path')
    parser.add_argument('--test_noisy_data_path', default='/SSD/training_set_canteenFB/Data_for_SEGAN/testset_noisy', type=str, help='test noisy data path')
    parser.add_argument('--test_data_path', default='/SSD/training_set_canteenFB/Data_for_SEGAN/preproc/serialized_test_data', type=str, help='test data path')
    parser.add_argument('--output_data_path', default='/SSD/training_set_canteenFB/Data_for_SEGAN/output/Inference', type=str, help='output data path')
    parser.add_argument('--model_save_path', default='/home/PycharmProjects/pythonProject/Model_Current_Training/SEGAN', type=str, help='model save path')
    parser.add_argument('--model_weights_save_path', default='/home/PycharmProjects/pythonProject/Model_Current_Training/SEGAN/model_weights_saved', type=str, help='model weights save path')
    parser.add_argument('--qc_plot', default=False, type=bool, help='plot qc')

    return parser.parse_args()

def load_datasets(opt):
    print('loading the datasets for testing...')
    test_CleanFile_lists = sorted(glob.glob(os.path.join(opt.test_clean_data_path, '*')))
    test_NoisyFile_lists = sorted(glob.glob(os.path.join(opt.test_noisy_data_path, '*')))
    return test_CleanFile_lists, test_NoisyFile_lists

def define_and_load_model(opt, num_epochs):
    print('Defining the generator model...')
    generator = GeneratorSegan()

    print('get the latest weights...')
    for epoch in range(num_epochs, 0, -1):
        if os.path.exists(f'{opt.model_weights_save_path}/generator_weights_epoch_{epoch}.index'):
            generator.load_weights(f'{opt.model_weights_save_path}/generator_weights_epoch_{epoch}')
            print('load weights from epoch %d' % epoch)
            break
    return generator

def denoise_data(opt, generator, test_CleanFile_lists, test_NoisyFile_lists):
    print('denoising based on the generator model...')
    window_size = 2 ** 14  # about 1 second of samples
    sample_rate = 16000
    stride      = 0.5

    for noisy_sound_file,clean_sound_file in tqdm(zip(test_NoisyFile_lists,test_CleanFile_lists)):
        # check if the denoising has been applied
        if os.path.exists(f'{opt.output_data_path}/denoised_' + noisy_sound_file.split('/')[-1]):
            print('denoised_' + noisy_sound_file.split('/')[-1] + 'has been generated! move to next file...')
        else:
            # load the clean data
            noisy_sound, _ = librosa.load(noisy_sound_file, sr=sample_rate)
            clean_sound, _ = librosa.load(clean_sound_file, sr=sample_rate)

            # slice the data into small pieces
            Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)
            noisy_slices       = Waveform_generator.slice_signal(noisy_sound_file, window_size=window_size, stride=stride, sample_rate=sample_rate)
            clean_slices       = Waveform_generator.slice_signal(clean_sound_file, window_size=window_size, stride=stride, sample_rate=sample_rate)

            # denoise the data
            denoised_slices    = []
            clean_slices_qc    = []
            noisy_slices_qc    = []

            for noisy_slice,clean_slice in tqdm(zip(noisy_slices,clean_slices), desc='Apply audio enhancement to noisy data'):

                # get the preconditioned noisy data
                noisy_slice = emphasis(noisy_slice[np.newaxis, np.newaxis, :], emph_coeff=0.95)
                clean_slice = emphasis(clean_slice[np.newaxis, np.newaxis, :], emph_coeff=0.95)

                noisy_slice = tf.transpose(np.array(noisy_slice),[0, 2, 1])
                clean_slice = tf.transpose(np.array(clean_slice),[0, 2, 1])

                # generate random noise
                z = tf.random.normal(shape=(1, 8, 1024))

                # get the enhanced data
                denoised_slice = generator(noisy_slice, z)

                # remove the emphasis after the denoising
                denoised_slice = tf.transpose(np.array(denoised_slice),[0, 2, 1])
                clean_slice    = tf.transpose(np.array(clean_slice),[0, 2, 1])
                noisy_slice    = tf.transpose(np.array(noisy_slice),[0, 2, 1])

                denoised_slice = emphasis_librosa(denoised_slice.numpy(), emph_coeff=0.95, pre=False)
                clean_slice    = emphasis_librosa(clean_slice.numpy(), emph_coeff=0.95, pre=False)
                noisy_slice    = emphasis_librosa(noisy_slice.numpy(), emph_coeff=0.95, pre=False)

                # output the denoised data
                denoised_slices.append(denoised_slice)
                noisy_slices_qc.append(noisy_slice)
                clean_slices_qc.append(clean_slice)

                # QC on a single data batch
                if opt.qc_plot:
                    plt.figure()
                    plt.plot(np.reshape(noisy_slice,-1), label='noisy')
                    plt.plot(np.reshape(clean_slice,-1), label='clean')
                    plt.plot(np.reshape(denoised_slice,-1), label='denoised')
                    plt.grid()
                    plt.legend()
                    plt.title('QC on a single denoised slice')
                    plt.show()

            # save the denoised data
            Waveform_generator = WaveformDataGenerator(sample_rate, window_size, stride)
            denoised_sound     = Waveform_generator.reconstruct_from_slice(denoised_slices, window_size=window_size, stride=stride)
            clean_sound_qc     = Waveform_generator.reconstruct_from_slice(clean_slices_qc, window_size=window_size, stride=stride)
            noisy_sound_qc     = Waveform_generator.reconstruct_from_slice(noisy_slices_qc, window_size=window_size, stride=stride)

            if opt.qc_plot:
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
            print('writing denoised data to {}'.format(opt.output_data_path))
            sf.write(f'{opt.output_data_path}/denoised_' + noisy_sound_file.split('/')[-1], denoised_sound, sample_rate)
            sf.write(f'{opt.output_data_path}/clean_' + noisy_sound_file.split('/')[-1], clean_sound_qc, sample_rate)
            sf.write(f'{opt.output_data_path}/noisy_' + noisy_sound_file.split('/')[-1], noisy_sound_qc, sample_rate)

def main():
    opt = parse_arguments()
    test_CleanFile_lists, test_NoisyFile_lists = load_datasets(opt)
    generator = define_and_load_model(opt, opt.num_epochs)
    denoise_data(opt, generator, test_CleanFile_lists, test_NoisyFile_lists)

if __name__ == '__main__':
    main()