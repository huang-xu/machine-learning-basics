import numpy as np
from utils import logistic


class LogisticRegression:
    def __init__(self, learnging_rate=2e-2, threshold=0.5):
        self.threshold = threshold
        self.epsilon = 1e-5
        self.beta = learnging_rate
        self.stational = 50

    def __gradient(self, x:np.ndarray, t:np.ndarray):
        rss = t - logistic(np.matmul(x, self.w))
        return -np.mean(x * rss.reshape([-1 ,1]), axis=0)

    def __rss(self, X, t):
        ks = logistic(np.matmul(X, self.w)) - self.threshold
        return sum(t*ks < 0)/len(X)

    def fit(self, X, t):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(t) is not np.ndarray:
            t = np.array(t).reshape([-1, 1])
        self.w = np.random.normal(size=X.shape[1])
        self.loss = [self.__rss(X, t)]
        step_length = self.beta * self.__gradient(X, t)
        while np.sum(np.abs(step_length)) > self.epsilon:
            self.w -= step_length
            step_length = self.beta * self.__gradient(X, t)
            self.loss.append(self.__rss(X, t))
            if len(self.loss) > self.stational and abs(self.loss[-1] -self.loss[-self.stational]) < self.epsilon:
                break
        return self


    def predict(self, X):
        try:
            self.w
        except:
            raise ValueError("Model must be trained before test!")
        lscore = logistic(np.matmul(X, self.w))
        res = np.ones(len(X), dtype=np.int32)
        res[lscore<self.threshold] = -1
        return res


if __name__ == '__main__':
    from data_set import multi_normal
    #import matplotlib.pyplot as plt
    #from sklearn.linear_model import LogisticRegression
    np.random.seed(1004)
    data, y = multi_normal()
    lr = LogisticRegression().fit(data, y)
    predy = lr.predict(data)
    colors = ['red' if ay > 0 else 'blue' for ay in y]
    markers = ['x' if ay > 0 else 'x' for ay in predy]
    for a, b in zip(y, predy):
        print(a, b)
    #plt.plot(lr.loss)
    #plt.show()
    #for ad, ay, ap in zip(data, y, predy):
    #    plt.scatter(ad[0], ad[1], marker='.' if ap>0 else 'x', color='red' if ay>0 else 'blue')
    #plt.show()