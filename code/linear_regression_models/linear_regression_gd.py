import numpy as np
from data_set import sin_with_noise
from linear_regression_models.linear_regression import LinearRegression
import matplotlib.pyplot as plt
from preprocessing import Standardization


class LinearGredientDescent(LinearRegression):
    def __init__(self, beta=1e-2, epsilon=3e-5):
        self.beta = beta
        self.epsilon = epsilon
        super().__init__()

    def __gradient(self, x:np.ndarray, t:np.ndarray, w:np.ndarray):
        return x.T.dot(x.dot(w) - t)

    def __rss(self, x: np.ndarray, t: np.ndarray):
        return np.mean(np.square(self.test(x) - t))

    def fit(self, X, t):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(t) is not np.ndarray:
            t = np.array(t).reshape([-1, 1])
        self.w = np.random.normal(size=X.shape[1])
        self.loss = []
        step_length = self.beta * self.__gradient(X, t, self.w)
        while np.sum(np.abs(step_length)) > self.epsilon:
            self.w -= step_length
            step_length = self.beta * self.__gradient(X, t, self.w)
            self.loss.append(self.__rss(X, t))
        return self


if __name__ == "__main__":
    np.random.seed(3456)
    trainX, trainY, testX, testY = sin_with_noise(M=10)
    sd = Standardization(trainX[:, 2:])
    trainX[:, 2:] = sd.transform(trainX[:, 2:])
    testX[:, 2:] = sd.transform(testX[:, 2:])
    lo = LinearGredientDescent().fit(trainX, trainY)
    predy = lo.predict(testX)
    plt.scatter(trainX[:, 1], trainY, label='Training Data')
    plt.plot(testX[:, 1], testY, label='Test Data')
    plt.plot(testX[:, 1], predy, label='Predict')
    plt.legend()
    plt.show()
    plt.plot(lo.loss[500:])
    plt.show()