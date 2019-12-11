import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GaussianMixture(object):
    """
    Gaussian Mixture Model
    """
    def __init__(self, n_components=1, tol=1e-5, max_iter=100, verbose=False):
        self.n_components = n_components

        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _initialize_params(self, x):
        n_samples, _ = x.shape
        n_components = self.n_components

        resp = np.random.rand(n_samples, n_components)
        resp /= resp.sum(axis=1)[:, np.newaxis]

        weights, means, covars = self._estimate_gaussian_params(x, resp)
        return weights, means, covars

    def _estimate_gaussian_params(self, x, resp):
        n_samples, n_features = x.shape
        n_components = self.n_components 

        weights = resp.sum(axis=0)
        means = np.dot(resp.T, x) / weights[:, np.newaxis] # shape = (n_components, n_features)
        
        covars = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = x - means[k] # shape = (n_samples, n_features)
            covars[k] = np.dot(resp[:, k] * diff.T, diff) / weights[k] # shape = (n_features, n_features)

        return weights, means, covars

    def _e_step(self, x):
        n_samples, _ = x.shape
        n_components = self.n_components

        resp = np.empty((n_samples, n_components))

        for k, (weight, mu, sigma) in enumerate(zip(self.weights_ , self.means_, self.covars_)):
            prob = multivariate_normal(mean=mu, cov=sigma).pdf(x)
            resp[:, k] = prob * weight

        # Normalization
        resp /= resp.sum(axis=1)[:, np.newaxis]
        return resp

    def _m_step(self, x, resp):
        return self._estimate_gaussian_params(x, resp)

    def _compute_lower_bound(self, resp):
        return np.mean(np.log(resp).sum(axis=1))

    def fit(self, x):
        self.weights_, self.means_, self.covars_ = self._initialize_params(x)
        
        iter = 1
        prev_lower_bound = 0.0

        while True:
            # E step, to estimate responsibilities
            resp = self._e_step(x)
            # M step, to evaluate params
            self.weights_, self.means_, self.covars_ = self._m_step(x, resp)
            
            # Compute average resp
            lower_bound = self._compute_lower_bound(resp)
            change = np.abs(lower_bound - prev_lower_bound)
            if self.verbose: 
                print("Iter: %d| lower_bound = %.3f| change = %.3f" % (iter, lower_bound,change))
                print("Means:\n", self.means_)
                print()
            if change < self.tol: break
            prev_lower_bound = lower_bound

            # Check iterations
            iter += 1
            if iter >= self.max_iter: break

    def predict_proba(self, x):
        return self._e_step(x)

    def predict(self, x):
        return self._e_step(x).argmax(axis=1)


if __name__ == '__main__':
    n_samples = 200

    mu1 = np.array([0, 1])
    mu2 = np.array([2, 1])
    cov1 = np.mat("0.3 0;0 0.1")
    cov2 = np.mat("0.2 0;0 0.3")

    breakpoint = n_samples // 2

    samples = np.empty((n_samples, 2))
    samples[:breakpoint, :] = np.random.multivariate_normal(mean=mu1, cov=cov1, size=breakpoint)
    samples[breakpoint:, :] = np.random.multivariate_normal(mean=mu2, cov=cov2, size=n_samples-breakpoint)

    part1 = samples[:breakpoint]
    part2 = samples[breakpoint:]


    # Fit a GMM with two components
    clf = GaussianMixture(n_components=2, max_iter=5000, verbose=True)
    clf.fit(samples)

    pred = clf.predict(samples)

    # Plot
    pred_part1 = samples[pred == 0]
    pred_part2 = samples[pred == 1]

    plt.subplot(2, 1, 1)
    plt.plot(part1[:, 0], part1[:, 1], 'bo')
    plt.plot(part2[:, 0], part2[:, 1], 'rs')
    
    plt.subplot(2, 1, 2)
    plt.plot(pred_part1[:, 0], pred_part1[:, 1], 'bo')
    plt.plot(pred_part2[:, 0], pred_part2[:, 1], 'rs')

    plt.show()

    # Params
    print("Means:\n", clf.means_)
    print("Weights:\n", clf.weights_)
    print("Covs:\n", clf.covars_)