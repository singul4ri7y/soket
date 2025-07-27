from typing import Optional, List, Tuple
from soket import Tensor, stack
from soket.utils.data import Dataset
from math import ceil
import numpy as np
import warnings


def collate(sequence: List[object] | Tuple[object]) -> Tensor:
    if isinstance(sequence[0], Tensor):
        return stack(sequence)
    elif isinstance(sequence[0], (int, float, bool)):
        return Tensor(sequence)
    else:
        warnings.warn('Received sequence form dataset in dataloader is not Tensor. ' \
        'Try using soket.transforms.ToTensor() to transform samples to Tensor otherwise ' \
        'use the sequence at your own risk!')
        return sequence


class DataLoader:
    """
    DataLoader provides an iterable over the given dataset. For dataset
    shuffling, a random sampler is used.

    Args:

    """

    dataset: Dataset
    batch_size: int
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Total number of iterations possible
        self.max_iter = ceil(len(dataset) / batch_size)

        # `self.ordering` stores the order of indicies to progress the
        # iteration for mini-batch.

        # Sequential sampling
        if not shuffle:
            # Store the order of indicies 
            self.ordering = np.array_split(np.arange(len(dataset)), self.max_iter)
        
    # DataLoader is iterable
    def __iter__(self):
        # Initialize iterator index
        self.idx = 0

        # If shuffling, do random sampling.
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                self.max_iter)
        
        return self

    # Iterator logic
    def __next__(self):
        if self.idx >= self.max_iter:
            raise StopIteration()

        samples = [self.dataset[i] for i in self.ordering[self.idx]]
        
        # Update iteration index
        self.idx += 1

        # Samples might have data and the corresponding target.
        if isinstance(samples[0], (list, tuple)):
            # Collate both data and targets.
            data, targets = [x[0] for x in samples], [x[1] for x in samples]
            return collate(data), collate(targets)
        
        return collate(samples)