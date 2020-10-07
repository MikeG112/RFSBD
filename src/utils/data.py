# External imports
import numpy as np
import torch
from torch.utils.data import Dataset

class ArrayDataset(Dataset):
    '''
    Dataset instance that stores separate numpy/torch arrays, or lists, in
    a horizontally stacked fashion.
    '''
    def __init__(self, *arrays):
        # All arrays must be the same length
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(torch.from_numpy(np.array(array[index]))
                     for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]
