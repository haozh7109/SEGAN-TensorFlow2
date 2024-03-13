# -*- coding: utf-8 -*-
"""
Created on Tue. Jan 4 10:54:03 2022

@author: zhaoh
"""

import numpy as np
from tensorflow import keras
from Py_data_Preprocess_Hao import emphasis, emphasis_librosa

class DataLoader(keras.utils.Sequence):
    """
    Class for loading the data batches from the disk. (pytorch data loader in TF implementation)
    """

    def __init__(self, file_lists, batch_size=32):
        self.file_lists = file_lists
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_lists) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Get the filename lists
        filenames_list = self.file_lists

        # Get the beginning indices of the batch
        i = idx * self.batch_size
        batch_data_paths = filenames_list[i : i + self.batch_size]

        # Load the data batches (the preprocessed data has been saved as the textual format by numpy)
        x, y = [], []
        for path in batch_data_paths:
            x.append(np.load(path)[0].reshape(1, -1))  #  output dimension: (1, n_samples)
            y.append(np.load(path)[1].reshape(1, -1))

        # convert to numpy array
        x = np.array(x)
        y = np.array(y)

        # Apply necessary preprocessing to the data batches if needed (such as pre-emphasis etc.), input data dimension: [batch, channel, time]
        # x = emphasis(x, emph_coeff=0.95)
        # y = emphasis(y, emph_coeff=0.95)
        x = emphasis_librosa(x, emph_coeff=0.95)
        y = emphasis_librosa(y, emph_coeff=0.95)

        # reshape data back to dimension: [batch, time, channel]
        x = x.transpose((0, 2, 1))
        y = y.transpose((0, 2, 1))

        return x, y

    def reference_batch(self,batch_size=32):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """

        # Get the filename lists
        ref_filenames_list = np.random.choice(self.file_lists, batch_size)
        ref_batch = np.stack([np.load(file) for file in ref_filenames_list])

        # Apply necessary preprocessing to the data batches if needed (such as pre-emphasis etc.)
        # ref_batch = emphasis(ref_batch, emph_coeff=0.95)
        ref_batch = emphasis_librosa(ref_batch, emph_coeff=0.95)

        # reshape data back to dimension: [batch, time, channel]
        ref_batch = ref_batch.transpose((0, 2, 1))

        return ref_batch
