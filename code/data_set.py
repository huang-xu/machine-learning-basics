"""
A collection of dataset generation methods
"""
import numpy as np
import pandas as pd
from utils import pi_sin, pi_sin_noise, polynomial
#import matplotlib.pyplot as plt


def sf_ny_house(fn="../data/sf_ny_house.csv"):
    df = pd.read_csv(fn)
    d = df.values
    return d[:, 1:], d[:, 0]

def audiology_standardized_data(fn="../data/audiology.standardized.data.txt"):
    with open(fn, "r") as f:
        contents = f.readlines()
    data = np.array([arow.strip().split(",") for arow in contents])
    return data[:, :-2], data[:, -1]


def lin_with_noise(w=3, N=20, min_val=-10,
                   max_val=10, std=1.5):
    x = np.linspace(min_val, max_val, N)
    real_y = w * x
    train_y = real_y + np.random.normal(size=N, scale=std)
    train_x = np.array([np.ones(N), x]).T
    return train_x, train_y, train_x, real_y


def sin_with_noise(N=20, min_val=0, max_val=1, M=9, std=0.25):
    x = np.linspace(min_val, max_val, N)
    train_y = pi_sin_noise(x, std=std)
    train_X = polynomial(x, M)
    test_x = np.linspace(min_val, max_val, N*2)
    t = pi_sin(test_x)
    test_X = polynomial(test_x, M)
    return train_X, train_y, test_X, t


def multi_normal(N = 20, centers=[[1,0], [0,1]]):
    d = np.array([np.random.multivariate_normal(centers[k],
                                         cov=np.eye(len(centers[k]))/5,
                                         size=N) for k in range(len(centers))])
    data = d.reshape([-1, d.shape[2]])
    y = np.array([-1]*N + [1]*N)
    return data, y




if __name__ == '__main__':
    #trainx, trainy, testx, testy = sin_with_noise()
    #plt.scatter(trainx[:, 1], trainy, label='Training Data')
    #plt.plot(testx[:, 1], testy, label='Test Data')
    #plt.legend()
    #plt.show()
    data, targets = audiology_standardized_data()
    print(data)
    print(targets)