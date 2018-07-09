import numpy as np
from data_set import sin_with_noise
from data import housing_ch2
import matplotlib.pyplot as plt
import numpy.ma as ma


class LassoCD:
    def __init__(self, alpha=1e-3, threshold=1e-3):
        self.alpha = alpha
        self.threshold =threshold
        self.maxiter = 10000

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n, m = X.shape
        self.w = np.zeros([m, 1])
        update_dims = np.arange(0, len(self.w))
        n_iter = 0
        rss = np.sum(np.square(y - self.predict(X)))
        rss_delta = rss
        while rss_delta > self.threshold:
            for k in update_dims:
                Xk = X[:, k]
                pk = sum([X[i,k] * (y[i] - sum([X[i, j]* self.w[j]
                                                for j in range(m) if j!=k]))
                          for i in range(n)])
                zk = np.sum(Xk**2)
                margin = self.alpha/2
                if pk < -margin:
                    new_wk = (pk + margin) / zk
                elif zk > margin:
                    new_wk = (pk - margin) / zk
                else:
                    new_wk = 0
                self.w[k] = new_wk
            np.random.shuffle(update_dims)
            n_iter +=1
            rss_iter = np.sum(np.square(y - self.predict(X)))
            rss_delta = abs(rss - rss_iter)
            rss = rss_iter
            if n_iter > self.maxiter:
                break
        return self

    def predict(self, X):
        try:
            self.w
        except:
            raise ValueError("model must be trained before test!")
        return np.matmul(X, self.w)


if __name__ == "__main__":
    np.random.seed(3456)
    from sklearn.model_selection import train_test_split

    data = housing_ch2()
    data = data.values
    trainX, testX, trainY, testY = train_test_split(data[:, :-1], data[:, -1])
    lo = LassoCD(alpha=1e-4).fit(trainX, trainY)
    predy = lo.predict(testX)
    print(np.mean(np.abs(predy.reshape([-1])-testY.reshape([-1]))))
    print(testY)
    print(predy)
    """
    trainX, trainY, testX, testY = sin_with_noise(M=6)
    lo = LassoCD(alpha=1e-4).train(trainX, trainY)
    predy = lo.test(testX)
    plt.scatter(trainX[:, 1], trainY, label='Training Data')
    plt.plot(testX[:, 1], testY, label='Test Data')
    plt.plot(testX[:, 1], predy, label='Predict')
    plt.legend()
    plt.show()
    """


