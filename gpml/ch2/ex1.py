from itertools import product
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()

"""
Replicate the generation of random functions from Figure 2.2.
Use a regular (or random) grid of scalar inputs and the covariance function from eq. (2.16). 
Hints on how to generate random samples from multi-variate Gaussian distributions are given in section A.2.  
Invent some training data points, and make random draws from the resulting GP posterior using eq. (2.19).
"""

N = np.random.multivariate_normal

# eq 2.16: sq-exp covariance function
def sq_exp_cov(p: np.array, q: np.array) -> float:
    return np.exp(-0.5 * np.linalg.norm(p - q)**2)

def cov_matrix(cov_fn, x1: np.array, x2: np.array):
    n = len(x1)
    m = len(x2)
    return np.array([cov_fn(p, q) for (p, q) in product(x1, x2)]).reshape((n, m))

# sample prior 
x = np.linspace(-5, 5)
n = len(x)

np.random.seed(33)
mu0 = np.zeros(n)
K   = cov_matrix(sq_exp_cov, x, x)
prior_sample1 = N(mu0, K)
prior_sample2 = N(mu0, K)
prior_sample3 = N(mu0, K)

plt.plot(x, prior_sample1)
plt.plot(x, prior_sample2)
plt.plot(x, prior_sample3)
plt.fill_between(x, -2, 2, color = "darkgray", zorder = -1, alpha = 0.5)
plt.show()


# observe some data
data = [
    (-4, -2),
    (-3, 0),
    (-1, 1),
    (0, 2),
    (2, -1)
]

xo, f = zip(*data)

# sample posterior 

Kss = K
K__ = cov_matrix(sq_exp_cov, xo, xo) # gossip girl 
Knv = np.linalg.inv(K__)
Ks_ = cov_matrix(sq_exp_cov, x , xo)
K_s = cov_matrix(sq_exp_cov, xo, x )

mu = np.squeeze(np.array((Ks_ @ np.linalg.inv(K__)).dot(np.array(f).reshape((5,1)))))
K_postr  = Kss - Ks_ @ Knv @ K_s
sig2 = 2 * np.sqrt(K_postr.diagonal())

np.random.seed(33)
postr_sample1 = N(mu, K_postr)
postr_sample2 = N(mu, K_postr)
postr_sample3 = N(mu, K_postr)

plt.plot(x, postr_sample1)
plt.plot(x, postr_sample2)
plt.plot(x, postr_sample3)
plt.fill_between(x, mu - sig2, mu + sig2, color = "darkgray", zorder = -1, alpha = 0.5)
plt.show()