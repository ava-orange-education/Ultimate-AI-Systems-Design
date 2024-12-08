import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha, beta, gamma = 0.5, 0.3, 0.2  # Weights for utility components
quality_range = np.linspace(0.1, 1.0, 100)  # Quality levels
volume_range = np.linspace(1000, 100000, 100)  # Volume levels

# Marginal utility functions
def marginal_quality_utility(q, v):
    """Calculate marginal utility with respect to quality."""
    benefit = alpha * np.exp(-2 * q)  # Exponential diminishing returns
    cost = gamma * (0.1 * v * np.exp(2 * q) / 1e4)  # Exponential cost
    return benefit - cost

def marginal_volume_utility(q, v):
    """Calculate marginal utility with respect to volume."""
    benefit = beta * (1000.0 / (v * np.log(v + 100)))  # Logarithmic diminishing returns
    cost = gamma * (0.2 * q * np.log(v + 100) / 1e3)  # Logarithmic cost
    return benefit - cost

# Generate data for plots
MU_Q = [marginal_quality_utility(q, 50000) for q in quality_range]  # Fixed volume
MU_V = [marginal_volume_utility(0.5, v) for v in volume_range]  # Fixed quality

# Plot marginal utility curves
plt.figure(figsize=(12, 6))

# Marginal Quality Utility Plot
plt.subplot(1, 2, 1)
plt.plot(quality_range, MU_Q, label='MU_Q (Quality)', linewidth=2)
plt.title('Marginal Quality Utility', fontsize=14)
plt.xlabel('Quality (Q)', fontsize=12)
plt.ylabel('Marginal Utility (MU_Q)', fontsize=12)
plt.grid(True)
plt.legend()

# Marginal Volume Utility Plot
plt.subplot(1, 2, 2)
plt.plot(volume_range, MU_V, label='MU_V (Volume)', linewidth=2)
plt.title('Marginal Volume Utility', fontsize=14)
plt.xlabel('Volume (V)', fontsize=12)
plt.ylabel('Marginal Utility (MU_V)', fontsize=12)
plt.xscale('log')  # Logarithmic scale for volume
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
