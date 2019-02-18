from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from mvn import MVN
from utils import check_random_state

np.random.seed(123)

class IGMM():
    """Incremental Gaussian Mixture Model Class
    Based on "Incremental Learning of Multivariate Gaussian Mixture Models" by Engel and Heinen
    ref: https://github.com/alexbaucom17/MapLayer/blob/master/src/map_layer_ros/src/igmm.py
    """

    def __init__(self, n, sig_init, T_nov, v_min=5.0, sp_min=2.5, random_state=check_random_state(0)):
        """Initialize igmm
            
            Parameters:
                n - dimension of data
                sig_init - scale for initial covariance matrix
                T_nov - novelty constant 0 < T_nov <= 1, defines distance that new data point must be from
                    any other components in order to create a new component. Bigger means that more components will be
                    created and T_nov = 1 means that every point will have a new component. The authors used 0.01             
                v_min - how many updates must pass before checking if a component should be removed
                sp_min - minimum cumulative probability to keep a component
        """

        self.n_dim = n
        self.n_components = 0
        self.sig_init = sig_init
        self.T_nov = T_nov
        self.v_min = v_min
        self.sp_min = sp_min
        self.random_state=random_state


    def update(self,X):
        """Update igmm with new data point
            X - data point that should be n length numpy array
        """

        #size check
        if X.shape[0] != self.n_dim:
            raise ValueError("The length of the input vector must match the dimension of the mixture model")

        #initialize variables if this is the first data point
        if self.n_components == 0:

            self.n_components = 1
            self.means = [X] #component mean
            self.covariance = [self.sig_init**2*np.identity(self.n_dim)] #component covariance
            self.alpha = [1.0] #component weight/prior, called p(j) in the paper
            self.v = [1.0] #age of component j
            self.sp = [1.0] #accumulator of component j

        else:

            # compute the probabilty of belonging to each component and the threshold for that component
            p_x_given_j = np.zeros(self.n_components)
            thresh = np.zeros(self.n_components)
            for j in xrange(self.n_components):
                # get current component parameters
                Cj = self.covariance[j]
                uj = self.means[j]

                # compute probability
                a = (2. * np.pi) ** (self.n_dim / 2.0) * np.sqrt(np.linalg.norm(Cj))
                xu = (X - uj).reshape(-1, 1)
                b = -0.5 * np.dot(xu.T, np.dot(np.linalg.inv(Cj), xu))
                p_x_given_j[j] = 1. / a * np.exp(b)
                thresh[j] = self.T_nov / a

            #if we meet novelty criterion, then add new component
            if np.all(np.less(p_x_given_j,thresh)):

                self.n_components += 1
                self.means.append(X)  # component mean
                self.covariance.append(self.sig_init ** 2 * np.identity(self.n_dim))  # component covariance
                self.v.append(1.)  # age of component j
                self.sp.append(1.)  # accumulator of component j
                self.alpha.append(1./sum(self.sp))  # component weight/prior, called p(j) in the paper

            else:
                # otherwise we update all the components with the new data

                #use bayes rule to find p_j_given_x
                sum_pj = sum([p_x_given_j[j] * self.alpha[j] for j in xrange(self.n_components)])
                p_j_given_x = [p_x_given_j[j] * self.alpha[j]/sum_pj for j in xrange(self.n_components)]

                #do updates exactly as specified in the paper
                for j in xrange(self.n_components):
                    self.v[j] += 1
                    self.sp[j] += p_j_given_x[j]
                    ej = X - self.means[j]
                    wj = p_j_given_x[j]/self.sp[j]
                    d_uj = wj * ej
                    self.means[j] += d_uj
                    ej_star = X - self.means[j]
                    self.covariance[j] = (1.-wj)*self.covariance[j] + wj*np.outer(ej_star,ej_star) - np.outer(d_uj,d_uj)

            # check if any components need to be removed
            idx2remove = [j for j in xrange(self.n_components) if self.v[j] > self.v_min and self.sp[j] < self.sp_min]
            for j in sorted(idx2remove, reverse=True):
                self.n_components -= 1
                del self.means[j]
                del self.covariance[j]
                del self.v[j]
                del self.sp[j]
                del self.alpha[j]

            #normalize alpha values after all sp updates
            sum_sp = float(sum(self.sp))
            self.alpha = [self.sp[j]/sum_sp for j in xrange(self.n_components)]


    def condition(self, indices, x):
        """Conditional distribution over given indices.

        Parameters
        ----------
        indices : array, shape (n_new_features,)
            Indices of dimensions that we want to condition.

        x : array, shape (n_new_features,)
            Values of the features that we know.

        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        n_features = self.n_dim - len(indices)
        priors = np.empty(self.n_components)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariance[k],
                      random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            priors[k] = (self.alpha[k] *
                         mvn.marginalize(indices).to_probability_density(x))
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
        priors /= priors.sum()
        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

    def predict(self, indices, X):
        """Predict means of posteriors.

        Same as condition() but for multiple samples.

        Parameters
        ----------
        indices : array, shape (n_features_1,)
            Indices of dimensions that we want to condition.

        X : array, shape (n_samples, n_features_1)
            Values of the features that we know.

        Returns
        -------
        Y : array, shape (n_samples, n_features_2)
            Predicted means of missing values.
        """
        n_samples, n_features_1 = X.shape
        n_features_2 = self.n_dim - n_features_1
        Y = np.empty((n_samples, n_features_2))
        for n in range(n_samples):
            conditioned = self.condition(indices, X[n])
            Y[n] = conditioned.priors.dot(conditioned.means)
        return Y


    def get_most_likely(self):
        return self.means[np.argmax(self.sp)]
        
    def get_closest(self,xy):
        best_dist = float("Inf")
        best_mu = None
        for mu in self.means:
            dist = np.linalg.norm(xy-mu)
            if  dist < best_dist:
                best_dist = dist
                best_mu = mu
        return best_mu

    def get_means(self):
        return self.means
        
    def get_covs(self):
        return self.covariance

    def to_ellipses(self, factor=1.0):
            """Compute error ellipses.

            An error ellipse shows equiprobable points.

            Parameters
            ----------
            factor : float
                One means standard deviation.

            Returns
            -------
            ellipses : array, shape (n_components, 3)
                Parameters that describe the error ellipses of all components:
                angles, widths and heights.
            """

            res = []
            for k in range(self.n_components):
                mvn = MVN(mean=self.means[k], covariance=self.covariance[k],
                          random_state=self.random_state)
                res.append((self.means[k], mvn.to_ellipse(factor)))
            return res


