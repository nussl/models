import torch
import numpy as np
import gin

@gin.configurable
class DebugDataset(torch.utils.data.Dataset):
    """
    This dataset just wraps an existing dataset
    and always returns a random item. The length
    of the dataset is also set accordingly. This
    is to test whether a network can successfully
    overfit to a single item. That same item will 
    then be given for evaluation to make sure it gets
    good metrics.
    """
    def __init__(self, dataset, idx=None, dataset_length=20000, device='cuda'):
        if idx is None:
            idx = np.random.randint(len(dataset))
        
        self.dataset = dataset
        self.idx = idx
        self.dataset_length = dataset_length

        for attr in dir(dataset):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(dataset, attr))

    def __getitem__(self, i):
        return self.dataset[self.idx]

    def __len__(self):
        return self.dataset_length
