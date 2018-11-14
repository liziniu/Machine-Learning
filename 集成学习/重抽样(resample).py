import numpy as np


# 根据样本权重重新采样
def sample(x, p=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(x)
    if p is None:
        permutation = np.random.permutation(n)
        return x[permutation], permutation
    else:
        permutation = np.random.choice(n, size=n, p=p)
        return x[permutation], permutation

if __name__ == "__main__":
    from sklearn import datasets
    x, y = datasets.make_classification(n_samples=300, n_features=2, n_informative=2,
                                        n_redundant=0, n_repeated=0, n_clusters_per_class=2,
                                        n_classes=2, random_state=3)
    p = np.array([1/300] * 300)
    # print(p, "\n", p.shape)
    # sample(x, p=p)

