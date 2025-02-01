import numpy as np
import matplotlib.pyplot as plt

# Define the System dynamics
anchors = np.array([[10, 8], [2, 0], [0, 4]])
ny = len(anchors)
nx = 4
Q = 0.2 * np.eye(2)
R = 0.5 * np.eye(ny)
T_s = 0.01

A = np.block([[np.eye(2), T_s * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
Q_bar = np.block([[np.zeros((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), Q]])

# Dynamical System
def f(x, w):
    """System dynamics, x_t+ = f(x, w)"""
    s, v = x[:2], x[2:4]
    x_t_plus = np.concatenate((s + v * T_s, v + w[:2]))
    return x_t_plus

def h(x, v):
    """Output relationship, y = h(x, v)"""
    s = x[:2]
    y = np.empty(shape=(ny,))
    for i in range(ny):
        # Calculate the Euclidean distance using np.linalg.norm
        distance = np.linalg.norm(s - anchors[i])
        y[i] = distance + v[i]  # Add noise v[i]
    return y


def dh_x(x, _v):
    """
    Gradient of h wrt x
    """
    r = x[:2]  # Extract the position part of the state (s)
    dh = np.zeros(shape=(ny, nx))  # Initialize the Jacobian matrix with zeros
    for i in range(ny):  # Loop over each anchor point
        dh[i, :2] = (r - anchors[i]).T / np.linalg.norm(r - anchors[i])  # Compute gradient
    return dh

# Initialization
x_t = np.array([3, 5, 10, 10])  
x_t_pred = np.array([0, 0, 0, 0])
sigma_t_pred = 50 * np.eye(nx)
N = 50

x_true_cache = np.zeros((N, 2))
x_meas_cache = np.zeros((N - 1, 2))
x_true_cache[0, :] = x_t[:2]

for t in range(N - 1):
    # Obtain measurement
    v_t = np.random.multivariate_normal([0]*ny, R)
    y_t = h(x_t, v_t)

    # Measurement update
    C = dh_x(x_t, 0)
    Z = C @ sigma_t_pred @ C.T + R
    x_t_meas = x_t_pred + sigma_t_pred @ C.T @ np.linalg.solve(Z, y_t - h(x_t_pred, np.zeros(ny)))
    x_meas_cache[t, :] = x_t_meas[:2]
    sigma_t_meas = sigma_t_pred - sigma_t_pred @ C.T @ np.linalg.solve(Z, C @ sigma_t_pred)

    # Dynamics
    w_t = np.random.multivariate_normal([0, 0], Q)
    x_t = f(x_t, w_t)
    x_true_cache[t + 1, :] = x_t[:2]

    # Time update
    x_t_pred = f(x_t_meas, np.zeros(nx))
    sigma_t_pred = A @ sigma_t_meas @ A.T + Q_bar

# Plotting
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 15})
plt.plot(x_true_cache[:, 0], 'k', label='$x_t$ ')
plt.plot(x_meas_cache[:, 0], 'k--', label='$\hat{x}_{t|t}$ (Estimated Position(x))')
plt.plot(x_true_cache[:, 1], 'r', label='$y_t$ ')
plt.plot(x_meas_cache[:, 1], 'r--', label='$\hat{y}_{t|t}$ (Estimated Position (y))')
plt.legend()
plt.xlabel('Time instant, t')
plt.ylabel('x and y')
plt.savefig('ekf1.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 15})

# Extract true and estimated positions
x1_true, x2_true = x_true_cache[:, 0], x_true_cache[:, 1]  # True positions
x1_est, x2_est = x_meas_cache[:, 0], x_meas_cache[:, 1]    # Estimated positions

# Plot true trajectory
plt.plot(x1_true, x2_true, 'k-', label='True Position Trajectory')

# Plot estimated trajectory
plt.plot(x1_est, x2_est, 'b--', label='Estimated Position Trajectory')

# Plot starting point
plt.scatter(x1_true[0], x2_true[0], color='g', s=100, label='Start Position', edgecolors='black')

# Plot anchors
plt.scatter(anchors[:, 0], anchors[:, 1], color='r', s=100, marker='^', label='Anchors', edgecolors='black')

# Labels and legend
plt.xlabel("$x_1$ (Position along X-axis)")
plt.ylabel("$x_2$ (Position along Y-axis)")
plt.title("True vs. Estimated Trajectory of the System")
plt.legend()
plt.grid()
plt.axis("equal")  # Ensures equal scaling for both axes
plt.show()

