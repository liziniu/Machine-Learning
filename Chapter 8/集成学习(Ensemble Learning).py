from decision_tree import DecisionTree
from resample import sample
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


"""
最近写的重抽样的决策树稍微有问题，回来再考虑吧...
"""


class AdaBoost:
    def __init__(
            self,
    ):
        self.n = None           # 样本总数
        self.m = None           # 学习机的数量
        self.data = None        # [x, y]
        self.ws = []            # 每个样本weight
        self.epsilons = []      # 每次的误差
        self.alphas = []        # 每次的正确率
        self.trees = []         # 学习机
        self.accs = []          # 每次学习的准确率
        pass

    def train(self, x, y, m=1, seed=3):
        """

        x, y = datasets.make_classification(n_samples=300, n_features=10, n_informative=8,
                                        n_redundant=2, n_repeated=0, n_clusters_per_class=2,
                                        n_classes=2, random_state=1)
        """
        n = len(x)
        self.n = n
        self.m = m
        w = np.array([1/n] * n)     # 初始化w

        for i_episode in range(1, m+1):
            self.ws.append(w)
            # 带权训练
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(x, y, sample_weight=w)
            predict = tree.predict(x)
            # 判断标准
            identity = (predict == y).astype(np.float32)
            epsilon = np.sum(w * (1 - identity)) / np.sum(w)
            acc = 1 - epsilon
            if acc <= 0.5:
                break
            self.epsilons.append(epsilon)
            self.accs.append(acc)
            self.trees.append(tree)
            # 更新alpha
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            self.alphas.append(alpha)
            # 5.更新w
            acc_ = np.mean((self.predict(x) == y).astype(np.float32))
            print("i:{0} --> accuracy: {1:.2f}, on all data:{2:.2f}".format(i_episode, acc, acc_))
            w = w * np.exp(-alpha * (y * predict))
            w = w / np.sum(w)
        print("训练结束，训练了{}个学习机".format(i_episode))

    def train_(self, x, y, m=10, seed=1):
        n = len(x)
        y_ = y.reshape(n, 1)
        self.n = n
        self.m = m
        self.data = np.hstack((x, y_))
        w = np.array([1/n] * n)     # 初始化w
        flag = False

        for i_episode in range(1, m+1):
            self.ws.append(w)
            # 抽样
            data, index = sample(self.data, p=w, random_state=seed)
            # 带权训练
            tree = DecisionTree()
            tree.build_tree(data)
            predict = tree.predict(data)
            # 判断标准
            identity = (predict == y).astype(np.float32)
            epsilon = np.sum(w[index] * (1 - identity)) / np.sum(w[index])
            acc = 1 - epsilon
            if acc <= 0.5:
                print("训练结束，训练了{}个学习机".format(i_episode-1))
                return
            self.epsilons.append(epsilon)
            self.accs.append(acc)
            self.trees.append(tree)
            # 更新alpha
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
            self.alphas.append(alpha)
            # 5.更新w
            acc_ = np.mean((self.predict(x) == y).astype(np.float32))
            print("i:{0} --> accuracy: {1:.2f}, on all data:{2:.2f}".format(i_episode, acc, acc_))
            predict_ = tree.predict(self.data)
            w = w * np.exp(-alpha * (y * predict_))
            w = w / np.sum(w)
        print("训练结束，训练了{}个学习机".format(i_episode))

    def predict(self, x):
        result = np.array([0.0]*len(x))
        m = len(self.trees)
        for i in range(m):
            tree = self.trees[i]
            alpha = self.alphas[i]
            result += alpha * tree.predict(x)
        return np.sign(result)

if __name__ == "__main__":
    x, y = datasets.make_classification(n_samples=300, n_features=4, n_informative=3,
                                        n_redundant=1, n_repeated=0, n_clusters_per_class=2,
                                        n_classes=2, random_state=10)
    n = len(x)
    y = 2*y - 1            # convert to {-1, 1}
    ada = AdaBoost()
    ada.train_(x, y, m=10)
    predict = ada.predict(x)
    print("accuracy: {}".format(np.mean((predict == y).astype(np.float32))))