import numpy as np

dna_bases = ["A", "C", "G", "T", "N"]

"""
There are four computational stages, in order:
1. convolution
2. rectification
3. pooling
4. neural network
"""


def score(sequence):
    # f(s) = network(pooling(rectification(convolution(s, M), b)), weights)
    pass


# Convolution stage
# M represents motif detectors, which is a (d, m, 4) matrix
def convolution(sequence, M):
    d = M.shape[0]
    m = M.shape[1]

    S = pad_sequence(sequence, m)  # (n+2m−2, 4) matrix
    X = np.zeros((len(sequence) + m - 1, d))  # (n+m-1, d) matrix

    for i in range(len(sequence) + m - 1):
        for k in range(M.shape[0]):
            sum = 0
            for j in range(m):
                for l in range(4):
                    sum += S[i + j, l] * M[k, j, l]
            X[i, k] = sum

    return X


def pad_sequence(sequence, m):
    assert m > 0
    base = ["A", "C", "G", "T", "N"]
    row_length = len(sequence) + 2 * (m - 1)
    S = np.zeros((row_length, 4))
    for i in range(0, row_length):
        for j in range(0, 4):
            if (i < m - 1) or (i > len(sequence) + m - 2) or (sequence[i - m + 1] == "N"):
                S[i, j] = 0.25
            elif sequence[i - m + 1] == base[j]:
                S[i, j] = 1
            else:
                S[i, j] = 0
    return S


# Rectification stage
# b represents the thresholds, with length d
def rectification(X, b):
    Y = np.zeros(X.shape)
    for i in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            Y[i, k] = max(0, X[i, k] - b[k])
    return Y


# Pooling stage
# Reminder Y is of shape (n+m-1, d)
def pooling(Y):
    d = Y.shape[1]
    z = np.zeros(d)
    for k in range(d):
        z[k] = max(list(Y[:, k]))
    return z


# Neural Network stage
# W represents the weights
def network():
    pass
