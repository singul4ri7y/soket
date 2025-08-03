from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod
from soket import Tensor

class Transform(ABC):
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError()


class Compose(Transform):
    """
    Similar to nn.Sequential. Usage:
        transforms.Compose(
            transforms.ToTensor,
            transforms.RandomCrop(padding=3)
        )
    """

    def __init__(self, transforms: List[Transform] | Tuple[Transform]):
        for t in transforms:
            assert isinstance(t, Transform)

        self.transforms = transforms

    def transform(self, X):
        for tform in self.transforms:
            X = tform(X)

        return X


class ToTensor(Transform):
    def transform(self, X: List) -> Tensor:
        return Tensor.from_numpy(X)
