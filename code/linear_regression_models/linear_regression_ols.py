import numpy as np
import matplotlib.pyplot as plt
from linear_regression_models.linear_regression import LinearRegression
from data import data_Boston_house_price, housing_ch2


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
    from sklearn.model_selection import train_test_split

    data = data_Boston_house_price()
    #data = data.values
    trainX, testX, trainY, testY = train_test_split(data[:, :-1], data[:, -1])
    lo = LinearOLS().fit(trainX, trainY)
    predy = lo.predict(testX)
    """
    plt.scatter(trainX[:, 2], trainY, label='Training Data')
    plt.scatter(testX[:, 2], testY, label='Test Data')
    plt.scatter(testX[:, 2], predy, label='Predict')
    plt.legend()
    plt.show()
    """
    t_p = lo.predict(trainX)
    print(np.mean(np.abs(t_p - trainY)))
    print(np.mean(np.abs(predy-testY)))
    #print(testY)
    #print(predy)
    #lo.draw()