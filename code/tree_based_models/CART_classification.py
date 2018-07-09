import numpy as np
from utils import weighted_conditional_gini, str2factor, draw_graph
from data_set import audiology_standardized_data, sf_ny_house


class Node:
    def __init__(self, node_id, level, data_ids,
                 forward_features,
                 test_feature=None, split_point=None,
                 left=None, right=None,
                 is_leaf=False, label=None, data_weights=None):
        self.node_id = node_id
        self.level = level
        self.left = left
        self.right = right
        self.data_ids = data_ids
        self.data_weights = data_weights
        self.test_feature = test_feature
        self.split_point = split_point
        self.is_leaf = is_leaf
        self.label = label
        self.forward_features = forward_features


class CARTClassifier:
    def __init__(self, max_depth=None, data_weights=None, max_n_cut_points=10):
        self.n_nodes = 0
        self.max_n_cut_points = max_n_cut_points
        self.max_depth = max_depth
        self.data_weights = data_weights

    @property
    def __node_id(self):
        self.n_nodes +=1
        return self.n_nodes

    def fit(self, X, y, data_weights=None, max_depth=None):
        if type(X) is not np.ndarray:
            X = np.array(X)
        self.data = X
        if type(y) is not np.ndarray:
            y = np.array(y)
        self.targets = y
        if max_depth is not None:
            self.max_depth = max_depth
        if data_weights is not None:
            self.data_weights = data_weights
        data_id_list = np.arange(len(y))
        all_features = np.arange(self.data.shape[1])
        self.root = Node(node_id=self.__node_id, level=0,
                         data_ids=data_id_list,
                         forward_features=all_features,
                         data_weights=self.data_weights)
        self.__tree_growth(self.root)
        return self

    def predict(self, X):
        return np.array([self.__tree_search(self.root, ax) for ax in X])

    def __feature_select(self, node:Node):
        targets = self.targets[node.data_ids]
        data = self.data[node.data_ids]
        test_feature, min_geni, split_p = node.forward_features[0], 1, 0
        for afeature in node.forward_features:
            feature_values = data[:, afeature]
            unique_vals = np.unique(feature_values)
            min_val, max_val = min(unique_vals), max(unique_vals)
            n_cuts = min(len(unique_vals)-1, self.max_n_cut_points)
            cut_points = np.linspace(min_val, max_val, n_cuts)
            for acutp in cut_points:
                idx = feature_values <= acutp
                geni = weighted_conditional_gini(targets, idx, w=node.data_weights)
                if geni < min_geni:
                    min_geni = geni
                    split_p = max(feature_values[idx])
                    test_feature = afeature
        return test_feature, split_p

    def __tree_growth(self, node:Node):
        node_targets = self.targets[node.data_ids]
        labels, freqs = np.unique(node_targets, return_counts=True)
        if len(node.forward_features) <= 0: # no more features
            node.label = labels[np.argmax(freqs)]
            node.is_leaf = True
            return
        if len(labels) <= 1: # all in one class
            node.label = labels[0]
            node.is_leaf = True
            return
        if self.max_depth is not None and node.level >= self.max_depth:
            node.label = labels[np.argmax(freqs)]
            node.is_leaf = True
            return

        test_feature, split_point = self.__feature_select(node)
        node.test_feature = test_feature
        node.split_point = split_point
        node_data = self.data[node.data_ids]
        lidx = node_data[:, test_feature] <= split_point
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

    def __tree_search(self, node:Node, x:np.ndarray):
        if node.is_leaf:
            return node.label
        feature_val = x[node.test_feature]
        if feature_val <= node.split_point:
            return self.__tree_search(node.left, x)
        else:
            return self.__tree_search(node.right, x)

    def draw(self, graph_name="tmp", file_name="tmp"):
        connects = []
        hints = {}
        self.__dfs(self.root, connects, hints)
        draw_graph(connects, hints, graph_name=graph_name, file_name=file_name)

    def __dfs(self, node:Node, connects, hints:dict):
        if node.is_leaf:
            hints[str(node.node_id)] = node.label
            return
        hints[str(node.node_id)] = node.test_feature
        lchild = node.left
        rchild = node.right
        connects.append([str(node.node_id), str(lchild.node_id), "<="+str(node.split_point)])
        connects.append([str(node.node_id), str(rchild.node_id), ">"+str(node.split_point)])

        self.__dfs(lchild, connects, hints)
        self.__dfs(rchild, connects, hints)

    def print(self):
        self.__dfs_print(self.root)

    def __dfs_print(self, node:Node):
        if node.is_leaf:
            print(" "*node.level, "level:%d"%node.level, " node ID:",
                  node.node_id, " label: ", node.label)
            return
        lchild = node.left
        rchild = node.right
        print(" "*node.level, "level:%d"%node.level,
                  " node ID:", node.node_id, " test feature: ",
                  node.test_feature, " child: %d, %d"% (lchild.node_id, rchild.node_id))
        self.__dfs_print(lchild)
        self.__dfs_print(rchild)



if __name__ == '__main__':
    # statistic_learning_methods_table51
    data = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                     [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]).T
    targets = data[:, -1]
    features = data[:, :-1]
    #print(targets)
    features, targets = audiology_standardized_data()
    features, targets = sf_ny_house()
    features = str2factor(features)
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY,  testY = train_test_split(features, targets, test_size=0.5)
    clf = CARTClassifier()
    clf.fit(trainX, trainY)
    predy= clf.predict(trainX)
    print(sum(trainY == predy)/len(trainY))
    predy = clf.predict(testX)
    print(sum(testY == predy) / len(testY))
    clf.draw()
