# Gaussian_Mixture
Gaussian_Mixture from scratch in Python using EM algorithm    

1. Randomly assign responsibility values for each item
2. Compute Gaussian params like means and so on
3. E step, using previous 'weights', 'means', 'covs' to evalute 'resp'
4. M step, using 'resp' to evaluate 'weights', 'means', 'covs'
5. Loop