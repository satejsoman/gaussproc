from itertools import product
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()
N = np.random.multivariate_normal

def wiener_cov(p, q):
    return min(p, q)

def brownian_bridge_cov(p, q):
    return min(p, q) - p*q

def cov_matrix(cov_fn, x1: np.array, x2: np.array):
    n = len(x1)
    m = len(x2)
    return np.array([cov_fn(p, q) for (p, q) in product(x1, x2)]).reshape((n, m))

x = np.linspace(0, 1)
n = len(x)
mu = np.zeros(n)
K = cov_matrix(brownian_bridge_cov, x, x)

np.random.seed(0)
for _ in range(100):
    plt.plot(x, N(mu, K))
plt.fill_between(x, -2*np.sqrt(K.diagonal()), 2*np.sqrt(K.diagonal()), color = "darkgray", zorder = -1, alpha = 0.5)
plt.show()