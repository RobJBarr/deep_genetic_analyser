import numpy as np


def generate_mutation_map(seq, model):
    """
    Given a sequence s = (s1,..,sn) of length n and
    DeepBind model, generate (4 x n) matrix
    """

    mutation_map = np.zeros(shape=(len(seq), 4))
    base = ['A', 'C', 'G', 'T']
    score = model.predict(seq)

    for i in range(mutation_map.shape[0]):
        for j in range(mutation_map.shape[1]):
            mutated_seq = seq[:i] + base[j] + seq[i + i:]
            new_score = model.predict(mutated_seq)
            mutation_map[i, j] = (new_score - score) * max(0, score, new_score)

    return mutation_map.T


def get_mutation_map(map):
    """
    Given a mutation map, generate an interaction logo
    """
    pass


