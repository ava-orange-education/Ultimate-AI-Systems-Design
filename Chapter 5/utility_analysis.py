# Retry with plotting directly
import numpy as np
import matplotlib.pyplot as plt

# Utility function parameters
alpha = 0.6  # weight for data quality
beta = 0.4   # weight for data volume
gamma = 0.1  # weight for cost
s = 50       # saturation point for diminishing returns
c = 0.02     # marginal cost per unit data volume

# Data simulation
volume = np.linspace(1, 100, 500)  # Data volume range
quality = 0.8  # Data quality constant

# Utility function
def utility_function(Q, V, alpha, beta, gamma, c):
    f_Q = Q  # Linear function for quality utility
    g_V = 1 - np.exp(-V / s)  # Sigmoid-like function for volume utility
    cost = gamma * V * c  # Cost as a function of volume
    return alpha * f_Q + beta * g_V - cost

# Marginal utility function
def marginal_utility(Q, V, s, c):
    return Q / (1 + np.exp(-V / s)) - c

# Calculate utility and marginal utility
utility = utility_function(quality, volume, alpha, beta, gamma, c)
marginal_utility_values = marginal_utility(quality, volume, s, c)

# Plotting
plt.figure(figsize=(12, 6))

# Utility function plot
plt.subplot(1, 2, 1)
plt.plot(volume, utility, label="Utility", color="blue")
plt.title("Utility Function vs. Data Volume")
plt.xlabel("Data Volume (V)")
plt.ylabel("Utility (U)")
plt.axvline(x=s, color="red", linestyle="--", label="Saturation Point")
plt.legend()
plt.grid(True)

# Marginal utility plot
plt.subplot(1, 2, 2)
plt.plot(volume, marginal_utility_values, label="Marginal Utility", color="green")
plt.title("Marginal Utility vs. Data Volume")
plt.xlabel("Data Volume (V)")
plt.ylabel("Marginal Utility (dU/dV)")
plt.axhline(y=0, color="red", linestyle="--", label="Zero Marginal Utility")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
