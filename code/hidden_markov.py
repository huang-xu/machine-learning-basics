import numpy as np


def markov_model():
    N, M = 4, 2
    pi = (np.ones(N) / N).reshape([-1, 1])
    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [0.4, 0.0, 0.6, 0.0],
                  [0.0, 0.4, 0.0, 0.6],
                  [0.0, 0.0, 0.5, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.3, 0.7],
                  [0.6, 0.4],
                  [0.8, 0.2]])
    return pi, A, B


def state_transfer(pi, A):
    return A.T.dot(pi)

def observe(pi, B):
    return B.T.dot(pi)


if __name__ == "__main__":
    pi_t, A, B = markov_model()
    data = [0, 0, 1, 1, 0]
    for k in range(5):
        O = observe(pi_t, B)
        print(O[data[k], 0])
        pi_t = state_transfer(pi_t, A)
