import numpy as np
import math
import random
import os
from constants import npy_regx
import re


class Loader:

    def __init__(self, source_dir, batch_size, split_fractions = [0.7, 0.3]):
        self.source_dir = source_dir
        filenames = os.listdir(source_dir)
        filenames = [filename for filename in filenames if re.match(npy_regx, filename)]
        self.file_paths = [os.path.join(source_dir, filename) for filename in filenames]

        self.batch_size = batch_size

        self.nbatches = int(math.floor(len(self.file_paths) / batch_size))

        self.file_paths = np.array(self.file_paths[:batch_size * self.nbatches])
        self.file_paths = np.split(self.file_paths, self.nbatches)

        self.ntrain = int(self.nbatches * split_fractions[0])
        self.nval = self.nbatches - self.ntrain

        self.split_sizes = [self.ntrain, self.nval]
        self.batch_ix = [0, 0]

        self.test_data = np.load(self.file_paths[0][0])
        self.height = self.test_data[0].shape[0]
        self.width = self.test_data[0].shape[1]
        self.depth = self.test_data[0].shape[2]

    def next_batch(self, split_index):
        index = self.batch_ix[split_index]
        if split_index == 1:
            index += self.ntrain

        batch_paths = self.file_paths[index]
        X = np.zeros(shape=(self.batch_size, self.height, self.width, self.depth), dtype=np.float32)
        y = np.zeros(shape=(self.batch_size, 3), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            Xy = np.load(path)
            X[i] = Xy[0]
            y[i] = Xy[1]

        self.batch_ix[split_index] = (self.batch_ix[split_index] + 1) % self.split_sizes[split_index]
        return X, y


if __name__ == '__main__':
    loader = Loader('../data/clustered/000', 3)
    print(loader.width, loader.height, loader.depth)
    X, y = loader.next_batch(0)
    X, y = loader.next_batch(0)


