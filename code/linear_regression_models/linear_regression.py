import numpy as np
from data_set import lin_with_noise
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError('The train method must be implemented!')

    def predict(self, X):
        try:
            self.w
        except:
            raise ValueError("Model must be trained before test!")
        return np.matmul(X, self.w)


if __name__ == "__main__":
    np.random.seed(3456)
    trainX, trainY, testX, testY = lin_with_noise()
    plt.scatter(trainX[:, 1], trainY)
    plt.plot(testX[:, 1], testY)
    from linear_regression_models.linear_regression_ols import LinearOLS
    predy = LinearOLS().train(trainX, trainY).test(testX)
    plt.plot(testX[:, 1], predy)
    plt.show()
    #lr = LinearRegression()
    #lr.train([],[])

