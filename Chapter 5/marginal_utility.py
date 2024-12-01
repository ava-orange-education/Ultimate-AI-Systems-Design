import numpy as np
import matplotlib.pyplot as plt

def marginal_utility(q, v, s, c):
    return q / (1 + np.exp(-v / s)) - c

# Parameters
q = 0.8  # Data quality
s = 50   # Saturation point
c = 0.05  # Marginal cost

# Generate volume values
volume = np.linspace(0, 200, 100)

# Calculate marginal utility
mu = marginal_utility(q, volume, s, c)

# Plot marginal utility curve
plt.plot(volume, mu)
plt.xlabel('Data Volume')
plt.ylabel('Marginal Utility')
plt.title('Marginal Utility Curve')
plt.grid(True)
plt.show()