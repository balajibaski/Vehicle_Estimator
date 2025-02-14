# -*- coding: utf-8 -*-
"""
Extended Kalman Filter for Beacon-based Position Tracking with Motion Transitions
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = np.pi
linear_velocity = 10  # Constant velocity (units/s)
r_A = 48              # Radius for first circular motion
r_B = 22              # Radius for second circular motion

# Transition times (calculated based on angular thresholds)
theta_1 = np.deg2rad(15)
theta_2 = np.deg2rad(79)
theta_B = np.deg2rad(140)
theta_3 = np.deg2rad(101)
theta_4 = np.deg2rad(165)
theta_5 = np.deg2rad(250)

transition_t_1 = theta_1 / (linear_velocity / r_A)
transition_t_2 = 111 / linear_velocity
transition_t_3 = theta_B / (linear_velocity / r_B) + transition_t_2
transition_t_4 = 111 / linear_velocity + transition_t_3 - transition_t_1
transition_t_5 = theta_5 / (linear_velocity / r_A) + transition_t_4
transition_times = [transition_t_1, transition_t_2, transition_t_3, transition_t_4, transition_t_5]

# Beacon positions (known)
beacons = np.array([[-7, 54], [-2, -54], [134, 1]], dtype=np.float64)

class ExtendedKalmanFilter:
    def __init__(self, dt, std_pos, std_theta, std_dist, beacons, transition_times, v, r_A, r_B):
        self.dt = dt
        self.beacons = beacons
        self.transition_times = transition_times
        self.v = v
        self.r_A = r_A
        self.r_B = r_B

        # State: [x, y, theta]
        self.x = np.zeros((3, 1), dtype=np.float64)

        # Process noise covariance
        self.Q = np.diag([std_pos**2, std_pos**2, std_theta**2])

        # Measurement noise covariance
        self.R = np.eye(beacons.shape[0]) * (std_dist**2)

        # State covariance
        self.P = np.eye(3) * 10  # Initial uncertainty

    def _get_phase(self, t):
        """Determine current motion phase based on time"""
        if t < self.transition_times[0]:
            return 'circular', self.v/self.r_A
        elif t < self.transition_times[1]:
            return 'linear', 0
        elif t < self.transition_times[2]:
            return 'circular', self.v/self.r_B
        elif t < self.transition_times[3]:
            return 'linear', 0
        elif t < self.transition_times[4]:
            return 'circular', self.v/self.r_A
        else:
            return 'linear', 0

    def predict(self, current_time):
        phase_type, omega = self._get_phase(current_time)
        theta = self.x[2, 0]

        # Update position
        self.x[0] += self.v * np.cos(theta) * self.dt
        self.x[1] += self.v * np.sin(theta) * self.dt

        # Update theta based on phase
        if phase_type == 'circular':
            self.x[2] += omega * self.dt

        # Jacobian of motion model
        F = np.array([
            [1, 0, -self.v * np.sin(theta) * self.dt],
            [0, 1,  self.v * np.cos(theta) * self.dt],
            [0, 0, 1]
        ], dtype=np.float64)

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        x_pred = self.x[0, 0]
        y_pred = self.x[1, 0]

        # Compute measurement Jacobian and predicted distances
        H = []
        h = []
        for beacon in self.beacons:
            dx = x_pred - beacon[0]
            dy = y_pred - beacon[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 1e-6:  # Avoid division by zero
                H.append([0.0, 0.0, 0.0])
            else:
                H.append([dx/distance, dy/distance, 0.0])

            h.append(distance)

        H = np.array(H, dtype=np.float64)
        h = np.array(h, dtype=np.float64).reshape(-1, 1)

        # Kalman update
        y = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

# Simulation parameters
total_t = 43
num_steps = 500
dt = total_t / num_steps
t = np.linspace(0, total_t, num_steps)

# Generate true trajectory
true_x = np.zeros_like(t)
true_y = np.zeros_like(t)

for i, ti in enumerate(t):
    if ti < transition_times[0]:
        theta = (linear_velocity / r_A) * ti
        true_x[i] = r_A * -np.cos(theta + pi/2)
        true_y[i] = r_A * np.sin(theta + pi/2)
    elif ti < transition_times[1]:
        dx = linear_velocity * np.cos(theta_1) * dt
        dy = -linear_velocity * np.sin(theta_1) * dt
        true_x[i] = true_x[i-1] + dx
        true_y[i] = true_y[i-1] + dy
    elif ti < transition_times[2]:
        theta = (linear_velocity / r_B) * (ti - transition_times[1])
        true_x[i] = 100 + r_B * np.cos(theta)
        true_y[i] = -r_B * np.sin(theta)
    elif ti < transition_times[3]:
        dx = -linear_velocity * np.cos(theta_1) * dt
        dy = -linear_velocity * np.sin(theta_1) * dt
        true_x[i] = true_x[i-1] + dx
        true_y[i] = true_y[i-1] + dy
    else:
        theta = (linear_velocity / r_A) * (ti - transition_times[3]) + 2*theta_1
        true_x[i] = r_A * np.cos(theta + pi/2)
        true_y[i] = -r_A * np.sin(theta + pi/2)

# Initialize EKF
ekf = ExtendedKalmanFilter(dt=dt, std_pos=0.1, std_theta=0.01, std_dist=1.5,
                          beacons=beacons, transition_times=transition_times,
                          v=linear_velocity, r_A=r_A, r_B=r_B)

# Initial state (matches true initial position)
ekf.x = np.array([[true_x[0]], [true_y[0]], [np.arctan2(true_y[1]-true_y[0], true_x[1]-true_x[0])]],
                dtype=np.float64)

# Storage for results
estimated_positions = []
residuals = []

# Simulation loop
for i in range(num_steps):
    # Generate measurement
    z = []
    for beacon in beacons:
        dx = true_x[i] - beacon[0]
        dy = true_y[i] - beacon[1]
        noise = np.random.normal(0, 1.5)
        z.append(np.sqrt(dx**2 + dy**2) + noise)
    z = np.array(z).reshape(-1, 1)

    # EKF steps
    ekf.predict(current_time=t[i])
    ekf.update(z)

    # Store results
    estimated_positions.append(ekf.x.copy())
    residuals.append(np.linalg.norm(ekf.x[:2] - np.array([[true_x[i]], [true_y[i]]])))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(true_x, true_y, label='True Path')
plt.scatter(beacons[:, 0], beacons[:, 1], c='r', marker='^', s=100, label='Beacons')
est = np.array(estimated_positions)[:, :2, 0]
plt.plot(est[:, 0], est[:, 1], '--', label='EKF Estimate')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('True Path vs EKF Estimation')
plt.legend()
plt.axis('equal')
plt.grid()

# Residual plot
plt.figure()
plt.plot(residuals)
plt.xlabel('Time Step')
plt.ylabel('Position Error')
plt.title('Estimation Error Over Time')
plt.grid()

plt.show()