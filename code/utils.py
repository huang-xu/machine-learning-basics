"""
A collection of utilities
"""
import numpy as np
from graphviz import Digraph


def draw_graph(data, hints:{}, graph_name="tmp", file_name="tmp"):
    u = Digraph(graph_name, file_name)
    u.attr('node', shape='box')
    for x, y, z in data:
        x += "\n " + str(hints[x]) if x in hints.keys() else ""
        y += "\n " + str(hints[y]) if y in hints.keys() else ""
        u.edge(x, y, label=z)
    u.view()
    return u


def weighted_entropy(x:np.ndarray, w:np.ndarray, log=np.log2):
    x = np.array(x)
    w = np.array(w)
    D = np.sum(w)
    points = np.unique(x)
    ps = np.array([np.sum(w[x == ap])/D for ap in points])
    return np.sum(-ps * log(ps))


def weighted_conditinonal_entropy(x:np.ndarray, condition:np.ndarray,
                                  w:np.ndarray, log=np.log2):
    if type(x) is not np.ndarray:
        x = np.array(x)
    uniq_cond = np.unique(condition)
    n = np.sum(w)
    cond_en = 0.0
    for a_cond in uniq_cond:
        idx = np.where(condition == a_cond)[0]
        p_i = np.sum(w[idx])/n
        cond_en += p_i * weighted_entropy(x[idx], w[idx])
    return cond_en


def weighted_information_gain(x, condition, w=None):
    if w is None:
        w = np.ones(len(x))/len(x)
    return weighted_entropy(x, w) - weighted_conditinonal_entropy(x, condition, w)


def weighted_information_gain_ratio(x, condition, w=None):
    if w is None:
        w = np.ones(len(x))/len(x)
    return (weighted_entropy(x, w) - weighted_conditinonal_entropy(x, condition, w))/weighted_entropy(condition, w)


def str2factor(data:np.ndarray):
    """
    string features (e.g., 'job', 'work') into numerical factors (e.g., 1,2,3)
    :param data:
    :return:
    """
    res_data = np.zeros(data.shape)
    for k in range(data.shape[1]):
        dk = data[:, k]
        mappings = {d:i for i, d in enumerate(np.unique(dk))}
        res_data[:, k] = [mappings[d] for d in dk]
    return res_data


def entropy(x, log=np.log2):
    """
    Entropy of a list
    >>> entropy([0, 0, 0, 1, 1, 1])
    1.0
    >>> entropy([0])
    0.0
    >>> entropy([2, 2, 2])
    0.0
    >>> entropy(['a', 'a', 'b'])
    0.9182958340544896

    :param x:
    :return:
    """
    ls, freq = np.unique(x, return_counts=True)
    ps = freq/sum(freq)
    return np.sum(-ps * log(ps))


def conditional_entropy(x, condition):
    """
    conditional entropy of x under condition
    :param x:
    :param condition:
    :return:
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    uniq_cond = np.unique(condition)
    n = len(condition)
    cond_en = 0.0
    for a_cond in uniq_cond:
        idx = np.where(condition == a_cond)[0]
        p_i = len(idx)/n
        cond_en += p_i * entropy(x[idx])
    return cond_en


def information_gain(x, condition):
    return entropy(x) - conditional_entropy(x, condition)


def information_gain_ratio(x, condition):
    return (entropy(x) - conditional_entropy(x, condition))/entropy(condition)


def sigmoid(x):
    if type(x) is not np.ndarray:
        x = np.array(x)
    return 1.0/(1.0 + np.exp(-x))


logistic = sigmoid


def d_sigmoid(x):
    h = sigmoid(x)
    return h*(1-h)


d_losigstic = d_sigmoid


def pi_sin(x):
    """
    y = 2*pi*sin(x), x \in [min_val, max_val] with N points
    :param x:
    :return: y

    >>> pi_sin(0)
    0.0
    >>> pi_sin([0,0.25])
    array([ 0.,  1.])
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    y = np.sin(2*np.pi*x)
    return y


def pi_sin_noise(x, std=0.3):
    """
    y = 2*pi*sin(x) + normal noise,
    x \in [min_val, max_val) with N points
    :param x:
    :return: x
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    y = np.sin(2*np.pi*x) + np.random.normal(0,std, size=len(x))
    return y


def polynomial(x, M):
    """
    Polynomial of x power to [1,2,..., M-1]
    :param x:
    :param M:
    :return: x**i, i \in [1,2,..., M-1]

    >>> polynomial(2,5)
    array([ 1,  2,  4,  8, 16], dtype=int32)
    >>> polynomial(3,5)
    array([ 1,  3,  9, 27, 81], dtype=int32)
    """
    if type(x) is not np.ndarray:
        x = np.array(x)
    return np.array([x**i for i in range(M)]).T


def gini(x):
    _, freq = np.unique(x, return_counts=True)
    pk = freq/np.sum(freq)
    return 1 - np.sum(pk**2)


def weighted_gini(x, w=None):
    if w is None:
        w = np.ones(len(x))/len(x)
    x = np.array(x)
    w = np.array(w)
    D = np.sum(w)
    points = np.unique(x)
    pk = np.array([np.sum(w[x == ap])/D for ap in points])
    return 1 - np.sum(pk**2)


def condition_gini(x, condition):
    """

    :param x:
    :param condition: boolean vector
    :return:
    """
    idx = condition
    D = len(x)
    D1 = sum(idx)
    return gini(x[idx]) * D1/D + gini(x[~idx]) * (D - D1) /D


def weighted_conditional_gini(x, condition, w=None):
    if w is None:
        w = np.ones(len(x))/len(x)
    if type(x) is not np.ndarray:
        x = np.array(x)
    idx = condition
    D = np.sum(w)
    D1 = sum(w[idx])
    return weighted_gini(x[idx], w[idx]) * D1 / D + \
           weighted_gini(x[~idx], w[~idx]) * (D - D1) / D

if __name__ == "__main__":
    data = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                     [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]).T
    targets = data[:, -1]
    features = data[:, :-1]
    for k in range(features.shape[1]):
        univals = np.unique(features[:, k])
        for aval in univals:
            print(condition_gini(targets, features[:, k]==aval), weighted_conditional_gini(targets, features[:, k]==aval))