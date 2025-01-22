import cupy as cp

# Create a random number generator
rng = cp.random.default_rng()

# Example array
arr = cp.array([10, 20, 30, 40, 50])

# Randomly sample 3 elements without replacement
sampled = rng.choice(arr, size=3, replace=False)

print(sampled)