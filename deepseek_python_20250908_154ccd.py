# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
# Set a random seed for reproducibility
np.random.seed(42)
# Define the parameters for the copula
num_samples = 1000  # Number of samples to generate
rho = 0.7           # Correlation coefficient between the two variables
# 1. Define the mean vector and covariance matrix for the multivariate normal
mean = np.array([0, 0])
covariance = np.array([[1, rho],
                       [rho, 1]])

# 2. Generate samples from the multivariate normal distribution
mv_normal_samples = multivariate_normal.rvs(mean=mean, cov=covariance, size=num_samples)

# 3. Transform these samples to uniform margins using the standard normal CDF (Phi)
# This is the key step that creates the copula.
u = norm.cdf(mv_normal_samples[:, 0]) # Transform first variable
v = norm.cdf(mv_normal_samples[:, 1]) # Transform second variable
copula_samples = np.column_stack((u, v))
# 4. Create the visualization
plt.figure(figsize=(8, 6))
plt.scatter(copula_samples[:, 0], copula_samples[:, 1], alpha=0.6, s=10)
plt.title('Samples from a Gaussian Copula (ρ = 0.7)')
plt.xlabel('Margin U₁ ~ Uniform(0,1)')
plt.ylabel('Margin U₂ ~ Uniform(0,1)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.show()