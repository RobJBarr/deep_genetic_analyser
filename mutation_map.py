import numpy as np
import torch

from util import read_file

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_mutation_map(seq):
    """
    Given a sequence s = (s1,..,sn) of length n and
    DeepBind model, generate (4 x n) matrix
    """

    mutation_map = np.zeros(shape=(len(seq), 4))
    base = ['A', 'C', 'G', 'T']

    model = read_file("part2_model.pickle").to(device)
    score = model.predict(seq)
    print('Score:' + str(score))

    for i in range(mutation_map.shape[0]):
        for j in range(mutation_map.shape[1]):
            mutated_seq = seq[:i] + base[j] + seq[i + 1:]
            print('Mutated Sequence:' + mutated_seq)
            new_score = model.predict(mutated_seq)
            print('New score:' + str(new_score))
            mutation_map[i, j] = (new_score - score) * max(0, score, new_score)

    return mutation_map.T


def get_mutation_map(map):
    """
    Given a mutation map, generate an interaction logo
    """
    pass
