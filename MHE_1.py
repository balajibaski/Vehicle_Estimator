import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# System Model
A = np.array([[1.0, 0.1],  # State transition matrix
              [0.0, 1.0]])

C = np.array([[1, 0],  # Measurement matrix
              [0, 1]])

Q = np.diag([0.1, 0.1])  # Process noise covariance
R = np.diag([0.05, 0.05])  # Measurement noise covariance

# Simulate True System
np.random.seed(42)
N = 50  # Number of time steps
x_true = np.zeros((2, N))
y_meas = np.zeros((2, N))

x_true[:, 0] = [1.0, 0.5]  # Initial state

for k in range(1, N):
    x_true[:, k] = A @ x_true[:, k-1] + np.random.multivariate_normal([0, 0], Q)
    y_meas[:, k] = C @ x_true[:, k] + np.random.multivariate_normal([0, 0], R)

# Moving Horizon Estimator (MHE)
HORIZON = 5  # MHE window size
x_est = np.zeros((2, N))

def mhe_objective(x_flat, y_meas_window):
    """ Cost function for MHE """
    x = x_flat.reshape((2, HORIZON + 1))  # Reshape to (2, horizon+1)
    cost = 0

    for i in range(HORIZON):
        # System dynamics penalty (process noise)
        cost += np.linalg.norm(x[:, i+1] - A @ x[:, i], ord=2)**2 / np.linalg.norm(Q)

        # Measurement noise penalty
        cost += np.linalg.norm(y_meas_window[:, i] - C @ x[:, i], ord=2)**2 / np.linalg.norm(R)

    return cost

for k in range(HORIZON, N):
    # Initial guess: Use previous measurements
    x0_guess = np.tile(y_meas[:, k-HORIZON:k+1], (1, 1)).flatten()

    # Solve optimization problem
    result = opt.minimize(mhe_objective, x0_guess, args=(y_meas[:, k-HORIZON:k],),
                          method="L-BFGS-B")

    if result.success:
        x_est[:, k] = result.x.reshape((2, HORIZON + 1))[:, -1]  # Take the last estimated state
    else:
        print(f"Optimization failed at time step {k}")

# Plot Results
plt.figure(figsize=(10, 4))
plt.plot(x_true[0, :], label="True x", linestyle="dashed")
plt.plot(x_est[0, :], label="Estimated x", linestyle="solid")
plt.scatter(range(N), y_meas[0, :], label="Measurements y", s=10, alpha=0.5)
plt.legend()
plt.title("MHE State Estimation for x")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(x_true[1, :], label="True z", linestyle="dashed")
plt.plot(x_est[1, :], label="Estimated z", linestyle="solid")
plt.scatter(range(N), y_meas[1, :], label="Measurements y", s=10, alpha=0.5)
plt.legend()
plt.title("MHE State Estimation for z")
plt.show()
