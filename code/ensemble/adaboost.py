import numpy as np
from linear_classification_models.logistic_regression import LogisticRegression
from utils import logistic, str2factor
from tree_based_models.decision_tree_id3 import DecisionTree
from tree_based_models.CART_classification import CARTClassifier
from data_set import sf_ny_house


class BasicClassifier(LogisticRegression):
    def __init__(self, data_w, learnging_rate=3e-2, threshold=0.5):
        super().__init__(learnging_rate, threshold)
        self.data_w = data_w

    def __gradient(self, x:np.ndarray, t:np.ndarray):
        data_w = self.data_w.reshape([-1, 1])
        rss = t - logistic(np.matmul(x, self.w))
        return -np.mean(x * rss.reshape([-1, 1] * data_w), axis=0)


class Adaboost:
    def __init__(self, threshold=0.5,
                 basic_classifier=DecisionTree,
                 basic_classifier_depth=2):
        self.basic_classifier = basic_classifier
        self.threshold = threshold
        self.basic_classifier_depth=basic_classifier_depth

    def fit(self, X, t, M=20):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(t) is not np.ndarray:
            t = np.array(t).reshape([-1, 1])
        N = len(t)
        dm = np.array([1.0/N]*N)
        self.alphams = []
        self.classifiers = []
        for i in range(M):
            a_clf = self.basic_classifier(max_depth=self.basic_classifier_depth,
                                          data_weights=dm).fit(X, t)
            res = a_clf.predict(X)
            em = dm[res != t].sum()
            if em > 0.5:
                continue
            alpham = 0.5 * np.log((1-em)/em)
            is_right = np.ones(N)
            is_right[res != t] = -1.0
            wmi_coeff = dm * np.exp(-alpham * is_right)
            dm = wmi_coeff/np.sum(wmi_coeff)
            self.classifiers.append(a_clf)
            self.alphams.append(alpham)
            error_n = sum(t!=self.predict(X))
            if error_n <= 0:
                print("%d classifiers is enough."%(i+1))
                break
        return self

    def predict(self, X):
        clfs = np.array([clf.predict(X)
                         for clf in
                         self.classifiers])
        alphams = np.array(self.alphams)
        res = []
        for instance in range(len(X)):
            ars = clfs[:, instance]
            uniq_class = np.unique(ars)
            max, max_class = 0, uniq_class[0]
            for aclass in uniq_class:
                wsum = np.sum(alphams[ars == aclass])
                if wsum > max:
                    max = wsum
                    max_class = aclass
            res.append(max_class)
        return res


if __name__=="__main__":
    data = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                     [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]).T
    targets = data[:, -1]
    features = data[:, :-1]

    features, targets = sf_ny_house()
    #features = str2factor(features)
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY,  testY = train_test_split(features, targets, test_size=0.5)
    ab = Adaboost(basic_classifier=CARTClassifier, basic_classifier_depth=3).fit(features, targets)
    ab.fit(trainX, trainY)
    predy= ab.predict(trainX)
    print(sum(trainY == predy)/len(trainY))
    predy = ab.predict(testX)
    print(sum(testY == predy) / len(testY))

    #for i, aclf in enumerate(ab.classifiers):
    #    aclf.draw(graph_name="n%d"%i)

#    lr = DecisionTree(max_depth=1).fit(features, targets).predict(features)
    #print(lr)