import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





def data_Boston_house_price(fn="./boston.txt"):
    with open(fn, "r") as f:
        f_contents = f.readlines()
    col_names, col_des = [], []
    for vnale_row in f_contents[7:21]:
        name, des = vnale_row.strip().split(" ", 1)
        col_names.append(name.strip())
        col_des.append(des.strip())
    data = np.array([feature_row.strip().split()
                     for feature_row in f_contents[22:]])
    return data, col_names, col_des


def esl_ch2(seed_size=10, data_size=100,
            seeds=np.array([[1, 0], [0, 1]])):
    """
    data set used in elements of statistical learning, chapter 2
    :return:
    """
    np.random.seed(3452)
    n_seeds = seeds.shape[1]
    cov = np.eye(n_seeds)
    data = []
    idxs = list(range(seed_size))
    for k in range(n_seeds):
        seed_d = np.random.multivariate_normal(seeds[:, k], cov/2, size=seed_size)
        plt.scatter(seed_d[:, 0], seed_d[:, 1])
        center_idx = np.random.choice(idxs, size=data_size, replace=True)
        data.append([np.random.multivariate_normal(seed_d[aidx], cov / 5, size=1)[0] for aidx in center_idx])
    plt.show()
    return np.array(data)


def housing_ch2(fn='./regression_X27.txt'):
    with open(fn, 'r') as f:
        contents = f.readlines()
    col_names = [arow.split(',')[0].strip() for arow in contents[1:13]]
    d = np.array([arow.split()[1:] for arow in contents[13:]], dtype=np.float)
    return pd.DataFrame(d, columns=col_names)

if __name__ == "__main__":
    pass