import numpy as np


class Standardization:
    def __init__(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        n, m = X.shape
        self.means = np.mean(X, axis=0).reshape([1, -1])
        self.sigmas = np.std(X, axis=0).reshape([1, -1])

    def transform(self, X):
        return (X - self.means)/self.sigmas



if __name__ == "__main__":
    x = [[1, 4, 3],
         [2, 5, 6],
         [3, 6, 10]]
    print(Standardization(x).transform(x))
