import numpy as np
from tree_based_models.CART_classification import CARTClassifier, Node
from data import housing_ch2, data_Boston_house_price


def weighted_conditional_varsum(x, condition, w):
    if w is None:
        w = np.ones(len(x))/len(x)
    if condition is None:
        condition = np.array([True]*len(x))
    lefts = x[condition]
    left_ws = w[condition]
    lmean = np.sum(lefts*left_ws)
    lvar = np.sum(left_ws*((lefts - lmean)**2))
    rights = x[~condition]
    right_ws = w[~condition]
    rmean = np.sum(right_ws*rights)
    rvar = np.sum(right_ws * ((rights - rmean) ** 2))
    return lvar+rvar


class CARTRegressor(CARTClassifier):
    def __init__(self, epsilon=1e-3, max_depth=None,
                 data_weights=None, max_n_cut_points=10):
        super().__init__(max_depth=max_depth, data_weights=data_weights,
                         max_n_cut_points=max_n_cut_points)
        self.epsilon = epsilon

    def __feature_select(self, node:Node):
        targets = self.targets[node.data_ids]
        data = self.data[node.data_ids]
        test_feature, split_p = node.forward_features[0], 0.0
        min_var = weighted_conditional_varsum(targets, None,
                                              w=node.data_weights)
        for afeature in node.forward_features:
            feature_values = data[:, afeature]
            min_val, max_val = min(feature_values), max(feature_values)
            n_cuts = min(len(np.unique(feature_values))-1, self.max_n_cut_points)
            cut_points = np.linspace(min_val, max_val, n_cuts)
            for acutp in cut_points:
                idx = feature_values <= acutp
                varsum = weighted_conditional_varsum(targets, idx, w=node.data_weights)
                if varsum < min_var:
                    min_var = varsum
                    split_p = max(feature_values[idx])
                    test_feature = afeature
        return test_feature, split_p

    def __tree_growth(self, node:Node):
        node_targets = self.targets[node.data_ids]
        node_var = weighted_conditional_varsum(node_targets, None,
                                              w=node.data_weights)
        node_value = np.mean(node_targets)
        if len(node.forward_features) <= 0: # no more features
            node.label = node_value
            node.is_leaf = True
            return
        if node_var < self.epsilon: # var is small
            node.label = node_value
            node.is_leaf = True
            return
        if self.max_depth is not None and node.level >= self.max_depth:
            node.label = node_value
            node.is_leaf = True
            return

        test_feature, split_point = self.__feature_select(node)
        node.test_feature = test_feature
        node.split_point = split_point
        lidx = self.data[node.data_ids, test_feature] <= split_point
        l_data_ids = node.data_ids[lidx]
        if self.data_weights is not None:
            l_weights = node.data_weights[lidx]
        else:
            l_weights = None
        ldata = self.data[l_data_ids]
        l_forwards = np.array([afeature for afeature in node.forward_features
                               if len(np.unique(ldata[:, afeature])) > 1])
        node.left = Node(self.__node_id, node.level+1, l_data_ids,
                         l_forwards, data_weights=l_weights)
        r_data_ids = node.data_ids[~lidx]
        rdata = self.data[r_data_ids]
        r_forwards = np.array([afeature for afeature in node.forward_features
                               if len(np.unique(rdata[:, afeature])) > 1])
        if self.data_weights is not None:
            r_weights = node.data_weights[~lidx]
        else:
            r_weights = None
        node.right = Node(self.__node_id, node.level+1, r_data_ids,
                          r_forwards, data_weights=r_weights)
        self.__tree_growth(node.left)
        self.__tree_growth(node.right)


if __name__ == "__main__":
    np.random.seed(3456)
    from sklearn.model_selection import train_test_split

    #data = housing_ch2()
    #data = data.values
    data = data_Boston_house_price()
    trainX, testX, trainY, testY = train_test_split(data[:, :-1], data[:, -1])
    lo = CARTRegressor(max_depth=8).fit(trainX, trainY)
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