import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# --------------------------- Constant Turn Rate Model (CTR) --------------------------- #
class ConstantTurnRateModel:
    def __init__(self, dt, turn_rate):
        """
        Initializes the constant turn rate model.
        :param dt: Time step
        :param turn_rate: Constant turn rate (rad/s)
        """
        self.dt = dt
        self.omega = turn_rate
        self.state = np.zeros(4)  # [x, vx, y, vy]

    def state_transition_matrix(self):
        """
        Returns the state transition matrix for constant turn rate motion.
        """
        dt = self.dt
        omega = self.omega

        if omega == 0:  # Straight-line motion
            return np.array([
                [1, dt,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  1, dt],
                [0,  0,  0,  1]
            ])

        sin_wdt = np.sin(omega * dt)
        cos_wdt = np.cos(omega * dt)

        return np.array([
            [1, sin_wdt / omega,  0, -(1 - cos_wdt) / omega],
            [0, cos_wdt,          0, -sin_wdt],
            [0, (1 - cos_wdt) / omega,  1, sin_wdt / omega],
            [0, sin_wdt,          0, cos_wdt]
        ])

    def update_state(self):
        """
        Updates the state based on the CTR dynamics.
        """
        F = self.state_transition_matrix()
        self.state = F @ self.state  # Matrix multiplication

    def set_initial_state(self, x, vx, y, vy):
        """
        Sets the initial state of the object.
        """
        self.state = np.array([x, vx, y, vy])

    def get_state(self):
        return self.state

# --------------------------- Simulation Setup --------------------------- #
dt = 1.0  # Time step
num_steps = 50  # Number of simulation steps
radius = 20  # Circular motion radius
turn_rate = 1 / radius  # Omega (rad/s) for constant radius

# Initial conditions
vx = 1.0
vy = 0.0
x0, y0 = radius, 0  # Start at (radius, 0)

# Initialize model
model = ConstantTurnRateModel(dt, turn_rate)
model.set_initial_state(x0, vx, y0, vy)

# Noise Covariances
Q = np.diag([0.1, 0.1])  # Process noise covariance
R = np.diag([0.05, 0.05])  # Measurement noise covariance

# True States and Measurements
x_true = np.zeros((2, num_steps))  # [x, y] positions
y_meas = np.zeros((2, num_steps))  # Noisy measurements

# Generate data
np.random.seed(42)
for k in range(num_steps):
    model.update_state()
    state = model.get_state()
    x_true[:, k] = [state[0], state[2]]  # Store (x, y)

    # Add noise to measurements
    y_meas[:, k] = x_true[:, k] + np.random.multivariate_normal([0, 0], R)

# --------------------------- Moving Horizon Estimator (MHE) --------------------------- #
HORIZON = 5  # MHE window size
x_est = np.zeros((2, num_steps))  # Estimated states [x, y]

def mhe_objective(x_flat, y_meas_window):
    """ Cost function for MHE """
    x = x_flat.reshape((2, HORIZON + 1))
    cost = 0

    for i in range(HORIZON):
        # System dynamics penalty (process noise)
        cost += np.linalg.norm(x[:, i+1] - x[:, i], ord=2)**2 / np.linalg.norm(Q)

        # Measurement noise penalty
        cost += np.linalg.norm(y_meas_window[:, i] - x[:, i], ord=2)**2 / np.linalg.norm(R)

    return cost

for k in range(HORIZON, num_steps):
    x0_guess = np.tile(y_meas[:, k-HORIZON:k+1], (1, 1)).flatten()

    # Solve MHE optimization problem
    result = opt.minimize(mhe_objective, x0_guess, args=(y_meas[:, k-HORIZON:k],),
                          method="L-BFGS-B")

    if result.success:
        x_est[:, k] = result.x.reshape((2, HORIZON + 1))[:, -1]  # Take last estimated state
    else:
        print(f"Optimization failed at time step {k}")

# --------------------------- Plot Results --------------------------- #
time = np.arange(num_steps)

# Top-Down View (XY Path)
plt.figure(figsize=(10, 5))
plt.plot(x_true[0, :], x_true[1, :], label="True Path", linestyle="dashed", color="black")
plt.scatter(y_meas[0, :], y_meas[1, :], label="Noisy Measurements", color="red", s=10, alpha=0.5)
plt.plot(x_est[0, :], x_est[1, :], label="MHE Estimated Path", linestyle="solid", color="blue")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Constant Turn Rate Model with MHE Estimation (Radius = 20)")
plt.legend()
plt.grid()
plt.axis("equal")  # Ensures circular appearance
plt.show()

# X Position vs. Time
plt.figure(figsize=(10, 4))
plt.plot(time, x_true[0, :], label="True X", linestyle="dashed", color="black")
plt.scatter(time, y_meas[0, :], label="Noisy X Measurements", color="red", s=10, alpha=0.5)
plt.plot(time, x_est[0, :], label="MHE Estimated X", linestyle="solid", color="blue")
plt.xlabel("Time Step")
plt.ylabel("X Position")
plt.title("X Position Over Time")
plt.legend()
plt.grid()
plt.show()

# Y Position vs. Time
plt.figure(figsize=(10, 4))
plt.plot(time, x_true[1, :], label="True Y", linestyle="dashed", color="black")
plt.scatter(time, y_meas[1, :], label="Noisy Y Measurements", color="red", s=10, alpha=0.5)
plt.plot(time, x_est[1, :], label="MHE Estimated Y", linestyle="solid", color="blue")
plt.xlabel("Time Step")
plt.ylabel("Y Position")
plt.title("Y Position Over Time")
plt.legend()
plt.grid()
plt.show()
