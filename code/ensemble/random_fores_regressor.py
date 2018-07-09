import numpy as np
from ensemble.random_forest_classification import RandomForest
from data import data_Boston_house_price
from tree_based_models.CART_regression import CARTRegressor


class RandomForestRegressor(RandomForest):
    def __init__(self, basic_model, n_estimators=10,
                 feature_proportion=0.4, max_depth=None):
        super().__init__(basic_model, n_estimators,
                 feature_proportion, max_depth=max_depth)

    def predict(self, X):
        raw_predicts = np.array([atree.predict(X[:, afeature_set])
                                 for atree, afeature_set
                                 in zip(self.trees, self.features)])
        return raw_predicts.mean(axis=0)


if __name__ == "__main__":
    np.random.seed(3456)
    from sklearn.model_selection import train_test_split

    #data = housing_ch2()
    #data = data.values
    data = data_Boston_house_price()
    trainX, testX, trainY, testY = train_test_split(data[:, :-1], data[:, -1])
    lo = RandomForestRegressor(basic_model=CARTRegressor,
                               n_estimators=15,
                               feature_proportion=0.5,
                               max_depth=7).fit(trainX, trainY)
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
    #for i, atree in enumerate(lo.trees):
    #    atree.draw("n%d"%i)

