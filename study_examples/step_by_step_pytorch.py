import numpy as np

# generate data

np.random.seed(42)

x = np.random.rand(100, 1)  # x is 100 * 1, 2D matrix
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

print()
