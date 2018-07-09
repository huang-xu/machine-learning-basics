import numpy as np
import matplotlib.pyplot as plt
from linear_regression_models.linear_regression_ols import LinearOLS
from data_set import sin_with_noise


class RidgeOLS(LinearOLS):
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(y) is not np.ndarray:
            y = np.array(y).reshape([-1, 1])
        m = X.shape[1]
        self.w = np.linalg.inv(X.T.dot(X) + self.alpha*np.eye(m)).dot(X.T.dot(y))
        return self


if __name__ == "__main__":
    trainX, trainY, testX, testY = sin_with_noise(N =20, M=11)
    lo = LinearOLS().fit(trainX, trainY)
    predy = lo.test(testX)
    plt.scatter(trainX[:, 1], trainY, label='Training Data')
    plt.plot(testX[:, 1], testY, label='Test Data')
    plt.plot(testX[:, 1], predy, label='Linear OLS Predict')
    ro = RidgeOLS(alpha=1e-2).fit(trainX, trainY)
    predy = ro.predict(testX)
    plt.plot(testX[:, 1], predy, label='Risge OLS Predict')
    plt.legend()
    plt.show()
