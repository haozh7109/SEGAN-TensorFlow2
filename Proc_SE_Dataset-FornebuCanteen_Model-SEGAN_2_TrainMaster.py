# -*- coding: utf-8 -*-
"""
Created on Mon. Jan 4 12:54:03 2022

@author: zhaoh
"""

import tensorflow as tf
import os
import argparse
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import soundfile as sf
import tqdm
import time
from Py_data_Preprocess_Hao import emphasis_librosa
from Py_data_loader_Hao import DataLoader
from Test_Customized_CRN_model import GeneratorSegan, DiscriminatorSegan, VirtualBatchNorm

if __name__ == '__main__':

    # --------------------------------------------------
    # 1.0 Set training parameters
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description='Train Audio Enhancement with SEGAN')
    parser.add_argument('--batch_size', default=120, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=50, type=int, help='train epochs number')
    parser.add_argument('--train_data_path', default='/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/preproc/serialized_train_data', type=str, help='train data path')
    parser.add_argument('--test_data_path', default='/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/preproc/serialized_test_data', type=str, help='test data path')
    parser.add_argument('--output_data_path', default='/TensorFlowSSD/training_set_canteenFornebu/Data_for_SEGAN/output', type=str, help='output data path')
    parser.add_argument('--model_save_path', default='/home/hao_listen/PycharmProjects/pythonProject/Model_Current_Training/SEGAN', type=str, help='model save path')
    parser.add_argument('--model_weights_save_path', default='/home/hao_listen/PycharmProjects/pythonProject/Model_Current_Training/SEGAN/model_weights_saved', type=str, help='model weights save path')
    parser.add_argument('--qc_plot', default=False, type=bool, help='plot qc')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    TRAIN_DATA_PATH = opt.train_data_path
    TEST_DATA_PATH = opt.test_data_path
    OUTPUT_DATA_PATH = opt.output_data_path
    MODEL_SAVE_PATH = opt.model_save_path
    MODEL_WEIGHTS_SAVE_PATH = opt.model_weights_save_path
    QC_PLOT = opt.qc_plot


    # --------------------------------------------------
    # 2.0 Load dataset for training
    # --------------------------------------------------
    print('loading the datasets for training...')

    # generate train, test file list
    train_file_lists = sorted(glob.glob(os.path.join(TRAIN_DATA_PATH, '*')))
    test_file_lists  = sorted(glob.glob(os.path.join(TEST_DATA_PATH, '*')))

    # split train set into training and validation set if needed
    val_samples = 1000
    random.Random(999).shuffle(train_file_lists)
    train_file_lists = train_file_lists[:-val_samples]
    valid_file_lists = train_file_lists[-val_samples:]

    # define the data generator
    train_data_loader = DataLoader(train_file_lists, BATCH_SIZE)
    valid_data_loader = DataLoader(valid_file_lists, BATCH_SIZE)
    test_data_loader  = DataLoader(test_file_lists, BATCH_SIZE)
    reference_data_loader = DataLoader(train_file_lists, BATCH_SIZE)

    # generate a reference batch for virtual batch normalization (VBN)
    ref_batch = reference_data_loader.reference_batch(BATCH_SIZE)

    # --------------------------------------------------
    # QC plots
    # --------------------------------------------------

    if QC_PLOT:
        # QC on a single data batch
        clean_single_batch, noise_single_batch = next(iter(train_data_loader))

        print(f'clean single batch has shape: {clean_single_batch.shape}')
        print(f'noise single batch has shape: {noise_single_batch.shape}')
        plt.figure()
        plt.plot(noise_single_batch[0, :, 0], label='noise')
        plt.plot(clean_single_batch[0, :, 0], label='clean')
        plt.grid()
        plt.legend()
        plt.title('QC on a single data batch')

        # QC the random selected reference data batch
        plt.figure()
        plt.plot(ref_batch[0, :, 1], label='noise')
        plt.plot(ref_batch[0, :, 0], label='clean')
        plt.grid()
        plt.legend()
        plt.title('QC on a random selected reference data batch')


    # --------------------------------------------------
    # 3.0 Define the model
    # --------------------------------------------------
    print('Defining the model...')
    generator = GeneratorSegan()
    discriminator = DiscriminatorSegan()

    print('Defining the optimizer...')
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    # --------------------------------------------------
    # 4.0 Training
    # --------------------------------------------------

    print('Start training...')
    for epoch in range(NUM_EPOCHS):
        print('=' * 100)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print('=' * 100)

        # # if the model has been trained previously, load the model
        # if os.path.exists(f'{MODEL_SAVE_PATH}/generator_epoch_{epoch + 1}') and os.path.exists(f'{MODEL_SAVE_PATH}/discriminator_epoch_{epoch + 1}'):
        #     generator     = tf.saved_model.load(f'{MODEL_SAVE_PATH}/generator_epoch_{epoch + 1}')
        #     discriminator = tf.saved_model.load(f'{MODEL_SAVE_PATH}/discriminator_epoch_{epoch + 1}')
        #     print(f'Loaded the model of epoch {epoch+1} successfully!')

        if os.path.exists(f'{MODEL_WEIGHTS_SAVE_PATH}/discriminator_weights_epoch_{epoch + 1}.index') and os.path.exists(f'{MODEL_WEIGHTS_SAVE_PATH}/generator_weights_epoch_{epoch + 1}.index'):

            generator.load_weights(f'{MODEL_WEIGHTS_SAVE_PATH}/generator_weights_epoch_{epoch + 1}').expect_partial()  # use expect_partial() to avoid warning message when loading the weights (https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
            discriminator.load_weights(f'{MODEL_WEIGHTS_SAVE_PATH}/discriminator_weights_epoch_{epoch + 1}').expect_partial()
            print(f'Loaded the model of epoch {epoch+1} successfully!')

        else:
            print(f'No model found, start training on epoch {epoch+1}...')

            epoch_start_time = time.time()

            for batch_idx, (clean_batch, noise_batch) in enumerate(train_data_loader):
                print(f'{batch_idx + 1}/{len(train_data_loader)} batch')

                # generate latent vector with normal distribution
                z = tf.random.normal(shape=(BATCH_SIZE, 8, 1024))
                z = tf.Variable(z, dtype=tf.float32)

                # generate a batch of reference data for VBN
                ref_batch = reference_data_loader.reference_batch(BATCH_SIZE)

                # # train the discriminator to recognize noisy audio as noisy, and clean audio as clean
                with tf.GradientTape() as tape_discriminator:

                    clean_batch_results = discriminator(tf.concat([clean_batch, noise_batch], axis=-1), ref_batch)
                    clean_loss = tf.reduce_mean((clean_batch_results - 1.0) ** 2)  # L2 loss - we want outputs to be 1 (real cases)

                    denoised_batch = generator(noise_batch, z)
                    denoised_batch_results = discriminator(tf.concat([denoised_batch, noise_batch], axis=-1), ref_batch)
                    noisy_loss = tf.reduce_mean((denoised_batch_results - 0) ** 2)  # L2 loss - we want outputs to be 0 (fake cases)

                    discriminator_loss = clean_loss + noisy_loss

                # print(f'clean_loss:{clean_loss}, noisy_loss:{noisy_loss} and Total discriminator_loss:{discriminator_loss}')

                # calculate the gradients of the discriminator
                discriminator_gradients = tape_discriminator.gradient(discriminator_loss, discriminator.trainable_variables)

                # update the discriminator
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


                # train the generator
                with tf.GradientTape() as generator_tape:

                    # generate a batch of denoised data
                    denoised_batch = generator(noise_batch, z)
                    denoised_batch_results = discriminator(tf.concat([denoised_batch, noise_batch], axis=-1), ref_batch)

                    # calculate the loss of the generator
                    generator_l2_loss = 0.5 * tf.reduce_mean((denoised_batch_results - 1) ** 2)  # L2 loss - we want the discriminator to judge that the generated denoised batch is close to the clean batch
                    generator_l1_loss = tf.reduce_mean(tf.abs(denoised_batch - clean_batch))     # L1 loss
                    generator_condition_loss = 100 * generator_l1_loss                           # condition loss
                    generator_loss = generator_l2_loss  + generator_condition_loss

                # print(f'generator_l2_loss:{generator_l2_loss}, generator_l1_loss:{generator_l1_loss}, generator_condition_loss:{generator_condition_loss} and Total generator_loss:{generator_loss}')

                # calculate the gradients of the generator
                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)

                # update the generator
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

                # print the progress of the training
                print(f'Epoch: {epoch + 1}/{NUM_EPOCHS}, Batch: {batch_idx + 1}/{len(train_data_loader)}, Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}')

            # calculate the time taken for the epoch
            epoch_end_time = time.time()
            print(f'Time taken for epoch {epoch + 1}: {epoch_end_time - epoch_start_time}')


            # # save the model with weights
            # tf.saved_model.save(generator,f'{MODEL_SAVE_PATH}/generator_epoch_{epoch + 1}')
            # tf.saved_model.save(discriminator,f'{MODEL_SAVE_PATH}/discriminator_epoch_{epoch + 1}')

            # save the mode weights only
            generator.save_weights(f'{MODEL_WEIGHTS_SAVE_PATH}/generator_weights_epoch_{epoch + 1}')
            discriminator.save_weights(f'{MODEL_WEIGHTS_SAVE_PATH}/discriminator_weights_epoch_{epoch + 1}')


            # Test the model after every epoch
            clean_batch, noise_batch = next(iter(test_data_loader))

            # generate latent vector with normal distribution
            z = tf.random.normal(shape=(BATCH_SIZE, 8, 1024))
            z = tf.Variable(z, dtype=tf.float32)

            # generate denoised data
            denoised_batch  = generator(noise_batch, z)

            # remove the emphasis
            # clean_batch     = emphasis(clean_batch.transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)
            # noise_batch     = emphasis(noise_batch.transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)
            # denoised_batch  = emphasis(denoised_batch.numpy().transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)
            clean_batch     = emphasis_librosa(clean_batch.transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)
            noise_batch     = emphasis_librosa(noise_batch.transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)
            denoised_batch  = emphasis_librosa(denoised_batch.numpy().transpose(0, 2, 1), emph_coeff=0.95, pre=False).transpose(0, 2, 1)


            # save the denoised data
            for idx in range(0,denoised_batch.shape[0],5):

                denoised_sample = denoised_batch[idx]
                clean_sample    = clean_batch[idx]
                noise_sample    = noise_batch[idx]

                if QC_PLOT:
                    plt.figure()
                    plt.plot(noise_sample, label='Noise')
                    plt.plot(clean_sample, label='Clean')
                    plt.plot(denoised_sample, label='Denoised')
                    plt.legend()
                    plt.show()

                sf.write(OUTPUT_DATA_PATH + '/test_' + str(idx) + '_clean_signal.wav', clean_sample, 16000)
                sf.write(OUTPUT_DATA_PATH + '/test_' + str(idx) + '_noisy_signal.wav', noise_sample, 16000)
                sf.write(OUTPUT_DATA_PATH + '/test_' + str(idx) + '_denoised_signal.wav', denoised_sample, 16000)
    print('End training...')




