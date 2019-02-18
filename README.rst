===
gmr
===

.. image:: https://api.travis-ci.org/AlexanderFabisch/gmr.png?branch=master
   :target: https://travis-ci.org/AlexanderFabisch/gmr
   :alt: Travis
.. image:: https://landscape.io/github/AlexanderFabisch/gmr/master/landscape.svg?style=flat
   :target: https://landscape.io/github/AlexanderFabisch/gmr/master
   :alt: Code Health

Gaussian Mixture Models (GMMs) (using EM and incremental) for clustering and regression in Python.

.. image:: https://github.com/mjm522/gmr/blob/master/gmm_igmm.png

Example
-------

Estimate GMM from samples and sample from GMM::

    from gmr import GMM

    gmm = GMM(n_components=3, random_state=random_state)
    gmm.from_samples(X)
    X_sampled = gmm.sample(100)


Guassian Mixture Regression Estimation Maximization::

      from gmr import GMM
   
      gmm = GMM(n_components=3, random_state=0)
      gmm.from_samples(X)
      Y_gmm = gmm.predict(np.array([0]), X_test[:, np.newaxis])

Guassian Mixture Regression Incremental Update::

       from gmr import IGMM

      igmm = IGMM(n=2, sig_init=1.1, T_nov=.1)
      for i in range(X.shape[0]):
          igmm.update(X[i,:])
      Y_igmm = igmm.predict(np.array([0]), X_test[:, np.newaxis])

How Does It Compare to scikit-learn?
------------------------------------

There is an implementation of Gaussian Mixture Models for clustering in
`scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html>`_
as well. Regression could not be easily integrated in the interface of
sklearn. That is the reason why I put the code in a separate repository.

Installation
------------

from source::

    sudo python setup.py install

.. _PyPi: https://pypi.python.org/pypi
