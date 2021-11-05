import csv
import gzip
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


# This file contains functions for parsing a gzip file to inputs for training

def pad_sequence(sequence, motif_len, kind='DNA'):
    rows = len(sequence) + 2 * motif_len - 2
    S = np.empty([rows, 4])
    base = 'ACGT' if kind == 'DNA' else 'ACGU'
    for i in range(rows):
        for j in range(4):
            if i - motif_len + 1 < len(sequence) and sequence[i - motif_len + 1] == 'N' or i < motif_len - 1 or i > len(
                    sequence) + motif_len - 2:
                S[i, j] = 0.25
            elif sequence[i - motif_len + 1] == base[j]:
                S[i, j] = 1
            else:
                S[i, j] = 0
    return np.transpose(S)


def shuffle(sequence):
    b = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d


def complement(sequence):
    dict_complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complement_sequence = [dict_complement[base] for base in sequence]
    return complement_sequence


def get_reverse_complement(sequence):
    sequence = list(sequence)
    sequence.reverse()
    return ''.join(complement(sequence))


def print_file(file):
    with gzip.open(file, 'rt') as data:
        next(data)
        reader = csv.reader(data, delimiter='\t')
        for row in reader:
            print(row)


def parse_file(file, motif_len=24, reverse_complement=False):
    training_set = []

    with gzip.open(file, 'rt') as data:
        next(data)
        reader = csv.reader(data, delimiter='\t')

        for row in reader:
            sequence = row[2]
            shuffled_sequence = shuffle(row[2])

            training_set.append([pad_sequence(sequence, motif_len), [1]])
            training_set.append([pad_sequence(shuffled_sequence, motif_len), [0]])

            if reverse_complement:
                training_set.append([pad_sequence(get_reverse_complement(sequence), motif_len), [1]])
                training_set.append([pad_sequence(shuffle(get_reverse_complement(sequence)), motif_len), [0]])

    # Shuffle the training set
    random.shuffle(training_set)
    size = int(len(training_set) / 3)

    # Partition the training set into 3 different training + validation sets
    first_valid, first_train = training_set[0: size], training_set[size:]
    second_valid, second_train = training_set[size: size + size], training_set[0:size] + training_set[size + size:]
    third_valid, third_train = training_set[size + size:], training_set[0: size + size]

    return first_train, first_valid, second_train, second_valid, third_train, third_valid, training_set
