import numpy as np
from tree_based_models.decision_tree_id3 import DecisionTree
from tree_based_models.CART_classification import CARTClassifier
from data_set import sf_ny_house, audiology_standardized_data
from utils import str2factor


class RandomForest:
    def __init__(self, basic_model, n_estimators=10,
                 feature_proportion=0.4, minimal_features=2,
                 max_depth=None):
        self.trees = []
        self.features = []
        self.feature_prop=feature_proportion
        self.n_trees = n_estimators
        self.basic_model = basic_model
        self.minimal_features = minimal_features
        self.max_depth = max_depth

    def fit(self, X, t):
        X = np.array(X)
        t = np.array(t)
        N, M = X.shape
        m = max(int(self.feature_prop*M), self.minimal_features)
        data_list, feature_list = np.arange(N), np.arange(M)
        for k in range(self.n_trees):
            cur_idx = np.random.choice(data_list, size=N, replace=True)
            cur_fx = np.random.choice(feature_list, size=m, replace=False)
            self.features.append(cur_fx)
            cur_data = X[cur_idx][:, cur_fx]
            cur_targets = t[cur_idx]
            self.trees.append(self.basic_model(max_depth=self.max_depth).fit(cur_data, cur_targets))
        return self

    def predict(self, X):
        raw_predicts = np.array([atree.predict(X[:, afeature_set])
                                 for atree, afeature_set
                                 in zip(self.trees, self.features)])
        res = []
        for k in range(raw_predicts.shape[1]):
            label, freqs = np.unique(raw_predicts[:, k], return_counts=True)
            res.append(label[np.argmax(freqs)])
        return res


if __name__ == '__main__':
    # statistic_learning_methods_table51
    data = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                     [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]).T
    targets = data[:, -1]
    features = data[:, :-1]

    features, targets = sf_ny_house()
    features = str2factor(features)
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY,  testY = train_test_split(features,
                                                     targets,
                                                     test_size=0.2)

    clf = RandomForest(CARTClassifier, n_estimators=15, feature_proportion=0.4)
    clf.fit(trainX, trainY)
    predy= clf.predict(trainX)
    print(sum(trainY == predy)/len(trainY))
    predy = clf.predict(testX)
    print(sum(testY == predy) / len(testY))
    #clf.draw()
    #for i, atree in enumerate(clf.trees):
    #    atree.draw("a%d"%i)
