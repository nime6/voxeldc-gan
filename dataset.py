import numpy as np
import glob
import os
import util
import config
import binvox_rw

class Dataset:

    def __init__(self, path):
        self.index_in_epoch = 0
        self.examples = np.array(glob.glob(path))
        self.num_examples = len(self.examples)
        np.random.shuffle(self.examples)
        print("dataset path:", path)
        print("number of examples:", self.num_examples)

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            np.random.shuffle(self.examples)
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples

        end = self.index_in_epoch
        return self.read_data(start, end)

    def read_data(self, start, end):
        data = []
        for fname in self.examples[start:end]:
            with open(fname, 'r', encoding="utf-8") as f:
                re = binvox_rw.read_as_coord_array(f, fix_coords=False)
                print(re.data, re.data.shape, re.dims, re.translate)
        return np.array(data)
