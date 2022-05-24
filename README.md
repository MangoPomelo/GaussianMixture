# Gaussian Mixture
GaussianMixture from scratch in Python using EM algorithm    

1. Randomly assign responsibility values to each item
2. Compute Gaussian params like means and etc
3. E step, using previous 'weights', 'means' and 'covs' to evalute 'responsibility'
4. M step, using 'responsibility' to evaluate 'weights', 'means', 'covs'
5. Loop while beyond threshold
