# -*- coding: utf-8 -*-
"""
Created on Mon. Jan 3 12:54:03 2022

@author: zhaoh
"""

import tensorflow as tf
import os
import random
import glob
import logging
import json
from Utility.Py_data_loader_Hao import DataLoader
from Utility.model import GeneratorSegan, DiscriminatorSegan, VirtualBatchNorm

# Set up logging
logging.basicConfig(filename='SEGAN_TrainMaster.log', level=logging.INFO)

# function for loading the job configuration
def load_config(config_file):
    """Load configuration from a file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# function for loading the dataset for training
def load_datasets(config):
    """Load datasets for training."""
    logging.info('Loading the datasets for training...')
    train_file_lists = sorted(glob.glob(os.path.join(config['TRAIN_DATA_PATH'], '*')))
    test_file_lists  = sorted(glob.glob(os.path.join(config['TEST_DATA_PATH'], '*')))

    val_samples = 1000
    random.Random(999).shuffle(train_file_lists)
    train_file_lists = train_file_lists[:-val_samples]
    valid_file_lists = train_file_lists[-val_samples:]

    train_data_loader = DataLoader(train_file_lists, config['BATCH_SIZE'])
    valid_data_loader = DataLoader(valid_file_lists, config['BATCH_SIZE'])
    test_data_loader  = DataLoader(test_file_lists, config['BATCH_SIZE'])
    reference_data_loader = DataLoader(train_file_lists, config['BATCH_SIZE'])

    ref_batch = reference_data_loader.reference_batch(config['BATCH_SIZE'])

    return train_data_loader, valid_data_loader, test_data_loader, reference_data_loader, ref_batch

# function for defining the model
def define_model():
    """Define the model."""
    logging.info('Defining the model...')
    generator = GeneratorSegan()
    discriminator = DiscriminatorSegan()

    logging.info('Defining the optimizer...')
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    return generator, discriminator, generator_optimizer, discriminator_optimizer

# function for training the model
def train_model(config, train_data_loader, valid_data_loader, test_data_loader, reference_data_loader, ref_batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """Train the model."""
    logging.info('Start training...')
    for epoch in range(config['NUM_EPOCHS']):
        logging.info('=' * 100)
        logging.info(f"Epoch {epoch + 1}/{config['NUM_EPOCHS']}")
        logging.info('=' * 100)

        if os.path.exists(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/discriminator_weights_epoch_{epoch + 1}.index") and os.path.exists(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/generator_weights_epoch_{epoch + 1}.index"):
            generator.load_weights(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/generator_weights_epoch_{epoch + 1}").expect_partial()
            discriminator.load_weights(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/discriminator_weights_epoch_{epoch + 1}").expect_partial()
            logging.info(f'Loaded the model of epoch {epoch+1} successfully!')
        else:
            logging.info(f'No model found, start training on epoch {epoch+1}...')

            for batch_idx, (clean_batch, noise_batch) in enumerate(train_data_loader):

                # print the progress of the training
                print(f"{batch_idx + 1}/{len(train_data_loader)} batch")

                # generate latent vector with normal distribution
                z = tf.random.normal(shape=(config['BATCH_SIZE'], 8, 1024))
                z = tf.Variable(z, dtype=tf.float32)

                # generate a batch of reference data for VBN
                ref_batch = reference_data_loader.reference_batch(config['BATCH_SIZE'])

                # # train the discriminator to recognize noisy audio as noisy, and clean audio as clean
                with tf.GradientTape() as tape_discriminator:

                    clean_batch_results = discriminator(tf.concat([clean_batch, noise_batch], axis=-1), ref_batch)
                    clean_loss = tf.reduce_mean((clean_batch_results - 1.0) ** 2)  # L2 loss - we want outputs to be 1 (real cases)

                    denoised_batch = generator(noise_batch, z)
                    denoised_batch_results = discriminator(tf.concat([denoised_batch, noise_batch], axis=-1), ref_batch)
                    noisy_loss = tf.reduce_mean((denoised_batch_results - 0) ** 2)  # L2 loss - we want outputs to be 0 (fake cases)

                    discriminator_loss = clean_loss + noisy_loss

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

                # calculate the gradients of the generator
                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)

                # update the generator
                generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

                # print the progress of the training
                print(f"Epoch: {epoch + 1}/{config['NUM_EPOCHS']}, Batch: {batch_idx + 1}/{len(train_data_loader)}, Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}")

            # save the model weights after each epoch
            generator.save_weights(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/generator_weights_epoch_{epoch + 1}")
            discriminator.save_weights(f"{config['MODEL_WEIGHTS_SAVE_PATH']}/discriminator_weights_epoch_{epoch + 1}")

    logging.info('Training completed!')

# Main function
def main():
    """Main function to run the script."""
    config = load_config('config.json')
    train_data_loader, valid_data_loader, test_data_loader, reference_data_loader, ref_batch = load_datasets(config)
    generator, discriminator, generator_optimizer, discriminator_optimizer = define_model()
    train_model(config, train_data_loader, valid_data_loader, test_data_loader, reference_data_loader, ref_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

if __name__ == '__main__':
    main()
