"""
=====================================
Linear Gaussian Models for Regression
=====================================

In this example, we use a MVN to approximate a linear function and a mixture
of MVNs to approximate a nonlinear function. We estimate p(x, y) first and
then we compute the conditional distribution p(y | x).
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from gmr.utils import check_random_state
from gmr import MVN, GMM, IGMM, plot_error_ellipses


random_state = check_random_state(0)

n_samples = 10
X = np.ndarray((n_samples, 2))
X[:, 0] = np.linspace(0, 2 * np.pi, n_samples)
X[:, 1] = 1 - 3 * X[:, 0] + random_state.randn(n_samples)

mvn = MVN(random_state=0)
mvn.from_samples(X)

X_test = np.linspace(0, 2 * np.pi, 100)
mean, covariance = mvn.predict(np.array([0]), X_test[:, np.newaxis])

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Linear: $p(Y | X) = \mathcal{N}(\mu_{Y|X}, \Sigma_{Y|X})$")
plt.scatter(X[:, 0], X[:, 1])
y = mean.ravel()
s = covariance.ravel()
plt.fill_between(X_test, y - s, y + s, alpha=0.2)
plt.plot(X_test, y, lw=2)

n_samples = 100
X = np.ndarray((n_samples, 2))
X[:, 0] = np.linspace(0, 2 * np.pi, n_samples)
X[:, 1] = np.sin(X[:, 0]) + random_state.randn(n_samples) * 0.1

gmm = GMM(n_components=3, random_state=0)
gmm.from_samples(X)
Y_gmm = gmm.predict(np.array([0]), X_test[:, np.newaxis])

igmm = IGMM(n=2, sig_init=1.1, T_nov=.1)

for i in range(X.shape[0]):
    igmm.update(X[i,:])

Y_igmm = igmm.predict(np.array([0]), X_test[:, np.newaxis])


plt.subplot(1, 3, 2)
plt.title("GMM: Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
          "\mathcal{N}_{k, Y|X}$")
plt.scatter(X[:, 0], X[:, 1])
plot_error_ellipses(plt.gca(), gmm, colors=["r", "g", "b"])
plt.plot(X_test, Y_gmm.ravel(), c="k", lw=2)

plt.subplot(1, 3, 3)
plt.title("IGMM: Mixture of Experts: $p(Y | X) = \Sigma_k \pi_{k, Y|X} "
          "\mathcal{N}_{k, Y|X}$")
plt.scatter(X[:, 0], X[:, 1])
plot_error_ellipses(plt.gca(), igmm, colors=["r", "g", "b", 'm'])
plt.plot(X_test, Y_igmm.ravel(), c="k", lw=2)

plt.show()
