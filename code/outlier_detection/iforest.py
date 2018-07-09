# try to implement iforest
import numpy as np





def cn(n):
    return 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n

class Node:
    def __init__(self, id_list, depth,
                 test_feature=None, split_point=None,
                 left=None, right=None):
        self.id_list = id_list
        self.depth = depth
        if len(self.id_list) <=1:
            self.is_leaf = True
        else:
            self.is_leaf = False
        self.left = left
        self.right = right
        self.test_feature = test_feature
        self.split_point = split_point

    @property
    def real_depth(self):
        if self.is_leaf and len(self.id_list) >1:
            return self.depth + cn(len(self.id_list))
        else:
            return self.depth


class Iforest:
    """
    Isolation Forest：简单实现
    """
    def __init__(self, n_estimators=100,
                 max_samples=256,
                 contamination=0.1,
                 replace=False):
        """
        可调整的几个参数
        :param n_estimators: 树数
        :param max_samples: 单颗树采用的最大训练样本数
        :param contamination: 训练数据中的异常样本
        :param replace: 采样方式，默认为可放回采样
        """
        self.n_trees = n_estimators
        self.max_samples = max_samples
        self.replace = replace
        self.contamination = contamination
        self.train_data_depth = {} # a dict of pointID: depth

    def fit(self, X, y=None):
        """
        训练
        :param X: 格式为 实例数*特征数
        :param y: 非监督，唯恐
        :return: 训练好的模型
        """
        self.data = X
        self.N = X.shape[0]
        self.max_samples = min(self.N, self.max_samples)
        self.max_depth = np.log2(self.max_samples)
        self.trees = []
        ############### 生成n_trees棵树
        id_list = np.arange(self.N)
        for tree_id in range(self.n_trees):
            id_sample = np.random.choice(id_list,
                                           size=self.max_samples,
                                           replace=self.replace)
            aroot = Node(id_sample, 0)
            self.tree_growth(aroot)
            self.trees.append(aroot)
        ###############
        depths = np.array([np.mean(val)
                           for val in self.train_data_depth.values()])
        anomaly_size = int(self.N * self.contamination)
        self.threshold = np.sort(self.score(depths))[-anomaly_size]
        return self

    def predict(self, X):
        """
        测试
        :param X: 格式为 实例数*特征数
        :return: 预测结果，1 表示正常，-1 表示不正常
        """
        xs_in_trees = np.array([self.search_tree(atree, ax)
                       for ax in X for atree in self.trees])
        xs_in_trees = xs_in_trees.reshape([X.shape[0], -1])
        hxs = xs_in_trees.mean(axis=1)
        scores = self.score(hxs)
        ret = np.ones(X.shape[0])
        ret[np.where(scores > self.threshold)[0]] = -1
        return ret

    def record_leaf(self, node:Node):
        """
        record the information of leaf nodes
        the information is stored in self.train_data_depth
        :param node:
        :return:
        """
        ids = node.id_list
        depth = node.real_depth
        for aid in ids:
            if aid not in self.train_data_depth.keys():
                self.train_data_depth[aid] = [depth]
            else:
                self.train_data_depth[aid].append(depth)

    def tree_growth(self, node:Node):
        """
        给定node，随机确定node的特征和分割点，分出两个子节点，并分别递归调用gen_tree
        :param node: 待生长的节点
        :return: 无
        """
        if node.is_leaf: #
            self.record_leaf(node)
            return
        if node.depth >= self.max_depth:
            node.is_leaf = True
            self.record_leaf(node)
            return

        node_data = self.data[node.id_list]
        # make sure not all points are identical
        maxlist, min_list= np.max(node_data, axis=0), np.min(node_data, axis=0)
        candidate_features = np.where(maxlist - min_list >0)[0]
        if len(candidate_features) <=0:
            node.is_leaf = True
            self.record_leaf(node)
            return
        # feature selection: to be improved
        tf_feature = np.random.choice(candidate_features, size=1)[0]
        tf_max, tf_min = maxlist[tf_feature], min_list[tf_feature]
        # split point : to be improved
        sp_point = np.random.uniform(tf_min, tf_max, size=1)[0]
        while sp_point <= tf_min:
            sp_point = np.random.uniform(tf_min, tf_max, size=1)[0]

        left_sub_mask = node_data[:, tf_feature] <= sp_point
        left_idlist = node.id_list[left_sub_mask]
        right_idlist = node.id_list[~left_sub_mask]
        left_node = Node(left_idlist, node.depth+1)
        right_node = Node(right_idlist, node.depth+1)
        node.test_feature = tf_feature
        node.split_point = sp_point
        node.left = left_node
        node.right = right_node
        self.tree_growth(left_node)
        self.tree_growth(right_node)

    def score(self, hxs):
        cn_val = cn(self.N)
        return np.power(2, -1*hxs/cn_val)

    def search_tree(self, root:Node, x):
        if root.is_leaf:
            return root.real_depth
        tf = root.test_feature
        sp = root.split_point
        if x[tf] <= sp:
            return self.search_tree(root.left, x)
        else:
            return self.search_tree(root.right, x)


#######################
def simple_data_v1(normal=100, abnormal = 20):
    rng = np.random.RandomState(42)
    # 正常数据
    X = 0.3 * rng.randn(normal, 2)
    X_train = np.r_[X + 2, X - 2]
    # 异常数据
    X_outliers = rng.uniform(low=-4, high=4, size=(abnormal, 2))
    return np.concatenate([X_train, X_outliers]), [1]*100 + [-1]*20


if __name__ == '__main__':
    x, y = simple_data_v1()
    clf = Iforest()
    clf.fit(x)
    pred = clf.predict(x)
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1], c=pred)
    plt.show()
