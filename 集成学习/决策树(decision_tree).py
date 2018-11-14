import numpy as np
from sklearn import datasets


def mode(x):
    result = {}
    for data in x:
        if data not in result.keys():
            result[data] = 1
        else:
            result[data] += 1
    return max(result, key=result.get)


def divide_set(x, index_f, value):
    try:
        set_1 = x[x[:, index_f].astype(np.float32) <= value]
        set_2 = x[x[:, index_f].astype(np.float32) >  value]
    except:
        set_1 = x[x[:, index_f] == value]
        set_2 = x[x[:, index_f] != value]

    return set_1, set_2


def unique_counts(x):
    result = {}
    y = x[:, -1]
    for label in y:
        if label not in result:
            result[label] = 1
        else:
            result[label] += 1
    return result


def entropy(x):
    ent = 0.0
    result = unique_counts(x)
    for label in result.keys():
        p = result[label] / len(x)
        ent += (-p) * np.log2(p)
    return ent


def build_node(x):
    n_features = x.shape[1] - 1
    ent = entropy(x)
    n = len(x)

    best_gain = 0.0
    best_feature = None
    best_res = None
    best_set = None

    for index_f in range(n_features):
        if isinstance(x[0, index_f], int) or isinstance(x[0, index_f], float):
            value = np.median(x[:, index_f])
            set_1, set_2 = divide_set(x, index_f, value)
        else:
            value = mode(x[:, index_f])
            set_1, set_2 = divide_set(x, index_f, value)
        n_1, n_2 = len(set_1), len(set_2)
        ent_1, ent_2 = entropy(set_1), entropy(set_2)
        gain = ent - n_1/n * ent_1 - n_2/n * ent_2
        if gain >= best_gain:
            best_gain = gain
            best_feature = (index_f, value)
            best_res = mode(set_1[:, -1]), mode(set_2[:, -1])
            best_set = (set_1, set_2)
    return best_feature, best_res, best_set


def predict(x, best_feature, best_res):
    n = len(x)
    # 增添index，方便后面输出
    index = np.arange(n).reshape(n, 1)
    x = np.hstack((x, index))
    index_f, value = best_feature
    # 按照最优feature 进行分割
    set_1, set_2 = divide_set(x, index_f, value)
    n_1, n_2 = len(set_1), len(set_2)
    # 分类结果
    res_1 = np.array([best_res[0]] * n_1).reshape(n_1, 1)
    res_2 = np.array([best_res[1]] * n_2).reshape(n_2, 1)
    # 将分类结果加入到set
    set_1 = np.hstack((set_1, res_1))
    set_2 = np.hstack((set_2, res_2))

    # 重新整理数据
    res = np.vstack((set_1, set_2))
    result = sorted(res, key=lambda x: float(x[-2]))   # 按照之前的index输出
    # print(result)
    return np.array(result)[:, -1]


class DecisionTree:
    def __init__(self):
        self.data = None
        self.best_feature = None
        self.best_res = None
        self.best_set = None

    def build_tree(self, data):
        self.data = data
        best_feature, best_res, best_set = build_node(self.data)
        self.best_feature = best_feature
        self.best_set = best_set
        self.best_res = best_res
        return best_feature, best_res, best_set

    def predict(self, x):
        return predict(x, self.best_feature, self.best_res)


if __name__ == "__main__":
    x, y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2,
                                        n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                                        n_classes=2, random_state=1)
    y = y[:, np.newaxis]
    data = np.hstack((x, y))
    best_feature, best_res, best_set = build_node(data)
    set_1, set_2 = best_set[0], best_set[1]
    pred = predict(data, best_feature, best_res)
    print("train accuracy: ", np.mean((pred==data[:, -1]).astype(np.float32)))