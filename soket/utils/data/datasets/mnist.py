from typing import Optional, Callable
from soket.utils.data import Dataset
from soket.transforms import Transform
import numpy as np
import gzip as gz
import struct

class MNIST(Dataset):
    """ Represents an MNIST dataset """

    def __init__(
        self,
        images_filename: str,
        labels_filename: str,
        transforms: Transform = None,
        target_transforms: Transform = None
    ):
        super().__init__(transforms)
        self.target_transforms = target_transforms

        with gz.open(images_filename, 'rb') as img_file:
            magic, num_samples, height, width = struct.unpack('>iiii', img_file.read(16))

            # 2051 colors, which is the magic of the image data file
            assert(magic == 2051)

            # Read rest of the file as numpy array.
            self.data = np.frombuffer(img_file.read(), dtype=np.uint8)    \
                .reshape(num_samples, height * width).astype(np.float32)
            
            # Every pixel value ranges from 0-255 (unsigned byte). Normalizing it
            # would mean dividing it by 255.
            self.data /= 255.
        
        with gz.open(labels_filename, 'rb') as label_file:
            magic, num_samples = struct.unpack('>ii', label_file.read(8))

            # Label data hardcoded magic is 2049
            assert(magic == 2049)

            self.targets = np.frombuffer(label_file.read(), dtype=np.uint8)
    
    def __getitem__(self, index: int | slice) -> object:
        img, target = self.data[index], self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        
        return img, target
    
    def __len__(self) -> int:
        """ Returns the number of samples. """
        return len(self.data)

    def apply_target_transforms(self, target):
        if self.target_transforms is not None:
            for tform in self.target_transforms:
                target = tform(target)
            
        return target