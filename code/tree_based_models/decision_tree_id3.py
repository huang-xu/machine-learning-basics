import numpy as np
from utils import draw_graph, weighted_information_gain
from data_set import audiology_standardized_data, sf_ny_house


class Node:
    def __init__(self, node_id, node_level, id_list, forward_features,
                 test_feature=None, is_leaf=False,
                 label=None):
        self.id_list = id_list
        self.is_leaf = is_leaf
        self.test_feature = test_feature
        self.forward_features = forward_features
        self.children = {}
        self.label = label
        self.id = node_id
        self.level = node_level


class DecisionTree:
    def __init__(self, max_depth=None, data_weights=None,
                 criterion=weighted_information_gain):
        self.node_account = 0
        self.criterion = criterion
        self.max_depth = max_depth
        if data_weights is not None:
            self.data_weights = np.array(data_weights)
        else:
            self.data_weights = None

    @property
    def __node_num(self):
        self.node_account += 1
        return self.node_account

    def __feature_selection(self, data, targets,
                            features, data_weights):
        max_inf_gain = -1.0
        max_feature = 0
        max_feature_col = data[:, features[0]]
        for afeature in features:
            feature_col = data[:, afeature]
            ffagain = self.criterion(targets, feature_col, data_weights)
            if ffagain > max_inf_gain:
                max_inf_gain = ffagain
                max_feature = afeature
                max_feature_col = feature_col
        return max_feature, max_feature_col

    def __tree_grow(self, node:Node, data_w):
        node_targets = self.targets[node.id_list] # C
        node_data = self.data[node.id_list]
        if len(np.unique(node_targets)) <= 1: # all in one class
            node.is_leaf = True
            node.label = node_targets[0]
            return
        if len(node.forward_features) <= 0: # no more features
            node.is_leaf = True
            labels, freq = np.unique(node_targets, return_counts=True)
            node.label = labels[np.argmax(freq)]
            return
        if self.max_depth is not None and node.level >= self.max_depth:
            node.is_leaf = True
            labels, freq = np.unique(node_targets, return_counts=True)
            node.label = labels[np.argmax(freq)]
            return

        forward_features = node.forward_features.copy()
        max_feature, max_feature_col = self.__feature_selection(node_data,
                                                                node_targets,
                                                                forward_features,
                                                                data_w)
        node.test_feature = max_feature
        forward_features.remove(max_feature)
        unique_fs = np.unique(max_feature_col)
        for auf in unique_fs:
            idx = np.where(max_feature_col == auf)[0]
            new_id_list = [node.id_list[aidx] for aidx in idx]
            if self.data_weights is not None:
                new_dataw = self.data_weights[new_id_list]
            else:
                new_dataw = None
            sub_node = Node(self.__node_num, node.level+1, new_id_list, forward_features)
            node.children[auf] = sub_node
            self.__tree_grow(sub_node, new_dataw)

    def fit(self, X, y):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if type(y) is not np.ndarray:
            y = np.array(y)
        self.data = X
        self.targets = y
        id_list = list(range(self.targets.shape[0]))
        feature_list = list(range(self.data.shape[1]))
        self.root = Node(self.__node_num, 0, id_list, feature_list)
        self.__tree_grow(self.root, self.data_weights)
        return self

    def __tree_search(self, x, node:Node):
        if node.is_leaf:
            return node.label
        fval = x[node.test_feature]
        return self.__tree_search(x, node.children[fval])

    def predict(self, X):
        return np.array([self.__tree_search(ax, self.root) for ax in X])

    def draw(self, graph_name="tmp", file_name="tmp"):
        connects = []
        hints = {}
        self.__dfs(self.root, connects, hints)
        draw_graph(connects, hints, graph_name=graph_name, file_name=file_name)

    def __dfs(self, node:Node, connects, hints:dict):
        if node.is_leaf:
            hints[str(node.id)] = node.label
            return
        hints[str(node.id)] = node.test_feature
        for key, val in node.children.items():
            connects.append([str(node.id), str(val.id), str(key)])
            self.__dfs(val, connects, hints)

    def print(self):
        self.__dfs_print(self.root, 0)

    def __dfs_print(self, node:Node, level):
        if node.is_leaf:
            print(" "*level, "level:%d"%level, " node ID:",
                  node.id, " label: ", node.label)
            return
        for key, val in node.children.items():
            print(" "*level, "level:%d"%level,
                  " node ID:", node.id, " test feature: ",
                  node.test_feature, " child: ",key," ID: %d"% val.id)
            self.__dfs_print(val, level+1)


if __name__ == '__main__':
    # statistic_learning_methods_table51
    data = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                     [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]).T
    targets = data[:, -1]
    features = data[:, :-1]

    features, targets = audiology_standardized_data()

    clf = DecisionTree()
    clf.fit(features, targets)
    clf.draw()
    predy = clf.predict(features)
    kk = [i for i in range(len(targets)) if targets[i] != predy[i] ]
    print(kk)
    #clf.print()
