import numpy as np
import matplotlib.pyplot as plt
from linear_regression_models.linear_regression import LinearRegression


class LinearOLS(LinearRegression):
    def __init__(self):
        super().__init__()

    def fit(self, X, t):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(t) is not np.ndarray:
            t = np.array(t).reshape([-1, 1])
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(t))
        return self


if __name__ == "__main__":
    np.random.seed(3456)
    trainX, trainY, testX, testY = sin_with_noise(M=10)
    lo = LinearOLS().fit(trainX, trainY)
    predy = lo.predict(testX)
    plt.scatter(trainX[:, 1], trainY, label='Training Data')
    plt.plot(testX[:, 1], testY, label='Test Data')
    plt.plot(testX[:, 1], predy, label='Predict')
    plt.legend()
    plt.show()