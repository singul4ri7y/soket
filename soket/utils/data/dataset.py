from typing import Optional, Callable
from abc import ABC, abstractmethod
from soket.transforms import Transform

class Dataset(ABC):
    """ An abstract class representing a `Dataset`.
    
    All subclasses should override - __getitem__() and __len__()
    """

    def __init__(
        self, 
        transforms: Transform = None
    ):
        self.transforms = transforms
    
    @abstractmethod
    def __getitem__(self, index: int | slice) -> object:
        """ Fetches a data sample from the dataset for a given key. """
        return NotImplementedError

    @abstractmethod
    def __len__():
        """ Returns the length of the dataset. """
        return NotImplementedError