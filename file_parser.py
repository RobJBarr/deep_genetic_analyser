import csv
import gzip
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

"""
Functions which allows parsing gzip file into tensors for training
pad_sequence() is implemented from an algorithm described in DeepBind supplementary notes
"""


class Initialise(Dataset):

    def __init__(self, dataset):
        self.x = torch.from_numpy(np.asarray([d[0] for d in dataset], dtype=np.float32))
        self.y = torch.from_numpy(np.asarray([d[1] for d in dataset], dtype=np.float32))
        self.len = len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len


def load_as_tensor(dataset, batch_size):
    return DataLoader(dataset=Initialise(dataset), batch_size=batch_size)


def parse_file(file_path, motif_len=24):
    """
    Partitions a given gzip file to lists which can all then be parsed as tensors

    :param file_path: str
    :param motif_len: int
    :return:
    """

    # For each sequence in the gzip file, store the padded sequence, and whether it has been shuffled
    training_set = []

    with gzip.open(file_path, 'rt') as data:
        next(data)
        reader = csv.reader(data, delimiter='\t')

        for row in reader:
            sequence = row[2]
            shuffled_sequence = shuffle(row[2])

            training_set.append([pad_sequence(sequence, motif_len), [1]])
            training_set.append([pad_sequence(shuffled_sequence, motif_len), [0]])

    # Shuffle the training set
    random.shuffle(training_set)

    # Partition the training set into 3 different training + validation sets
    size = int(len(training_set) / 3)
    first_valid, first_train = training_set[0: size], training_set[size:]
    second_valid, second_train = training_set[size: size + size], training_set[0:size] + training_set[size + size:]
    third_valid, third_train = training_set[size + size:], training_set[0: size + size]

    return first_train, first_valid, second_train, second_valid, third_train, third_valid


def pad_sequence(sequence, motif_len, kind='DNA'):
    rows = len(sequence) + 2 * motif_len - 2
    S = np.empty([rows, 4])
    base = ['A', 'C', 'G', 'T'] if kind == 'DNA' else ['A', 'C', 'G', 'U']

    for i in range(rows):
        for j in range(4):
            if i - motif_len + 1 < len(sequence) and sequence[i - motif_len + 1] == 'N' or \
                    i < motif_len - 1 or i > len(sequence) + motif_len - 2:
                S[i, j] = np.float32(0.25)
            elif sequence[i - motif_len + 1] == base[j]:
                S[i, j] = np.float32(1)
            else:
                S[i, j] = np.float32(0)
    return np.transpose(S)


def shuffle(sequence):
    b = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d
