# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 02:04:16 2025

@author: Conal
"""

import numpy as np
import matplotlib.pyplot as plt


#constants
pi = np.pi
#transition points (measured manually)
theta_1 = (15/180)*pi 
theta_2 = (79/180)*pi
theta_B = (140/180)*pi   #135 from compass      #Angle needs measured from B to calculate the correct timestep
theta_3 = (101/180)*pi 
theta_4 = (165/180)*pi
theta_5 = (250/180)*pi

x_1 = -48*np.cos(theta_1+pi/2); y_1 = 48*np.sin(theta_1+pi/2); x_2 = -111*np.cos(theta_2+pi/2); y_2 = 111*np.sin(theta_2+pi/2); x_3 = -111*np.cos(theta_3+pi/2); y_3 = 111*np.sin(theta_3+pi/2); x_4 = -48*np.cos(theta_4+pi/2); y_4 = 48*np.sin(theta_4+pi/2);

#sets the approximate points for transition
transition_points_x = []; transition_points_y = []
transition_points_x.append([x_1,x_2,x_3,x_4]); transition_points_y.append([y_1,y_2,y_3,y_4])

#Track shape and boundaries
r_A = 48             # Radius of circular motion
r_B = 22
total_t = 43   # Total simulation time
linear_velocity = 10  # Velocity after transition (units/s)
v = linear_velocity
omega_A = v/r_A         # Angular velocity (rad/s)
omega_B = v/r_B
#transition times:
transition_t_1 = theta_1/omega_A # Time at which transition starts
transition_t_2 = 111/v
transition_t_3 = theta_B/omega_B + transition_t_2 
transition_t_4 = 111/v + transition_t_3 - transition_t_1
transition_t_5 = theta_5/omega_A + transition_t_4 

"""Filter stuff"""
class KalmanFilter:
    def __init__(self, dt, std_acc, std_omega, std_pos, std_vel):
        self.dt = dt  # Time step
        
        # State vector: [x, y, v, theta]
        self.x = np.zeros((4, 1))
        
        # State transition model (non-linear, will be applied manually)
        self.F = np.eye(4)
        
        # Process noise covariance matrix
        self.Q = np.diag([std_acc**2, std_acc**2, std_omega**2, std_omega**2])
        
        # Measurement matrix (assume we can measure x and y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Measurement noise covariance matrix
        self.R = np.diag([std_pos**2, std_pos**2])
        
        # State covariance matrix
        self.P = np.eye(4) * 10  # High initial uncertainty
    
    def predict(self):
        v = self.x[2, 0]
        theta = self.x[3, 0]
        
        # Update state using CTRV motion model
        self.x[0, 0] += v * np.cos(theta) * self.dt
        self.x[1, 0] += v * np.sin(theta) * self.dt
        self.x[3, 0] += self.dt  # Constant angular velocity
        
        # Linearized transition matrix
        self.F = np.array([[1, 0, np.cos(theta) * self.dt, -v * np.sin(theta) * self.dt],
                           [0, 1, np.sin(theta) * self.dt,  v * np.cos(theta) * self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Predict state covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        y = z - (self.H @ self.x)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + (K @ y)  # Update state
        self.P = (np.eye(4) - K @ self.H) @ self.P  # Update covariance

# Simulate circular motion
Number_of_time_intervals = 500
dt = total_t/Number_of_time_intervals  # Time step

kf = KalmanFilter(dt, std_acc=0.1, std_omega=0.1, std_pos=0.5, std_vel=0.5)
kf.x = np.array([[10], [0], [1], [0]])  # Initial state (x, y, v, theta)

# Generate measurements
num_steps = 100
true_positions = []
measurements = []

"""Filter stuff^^^"""
# Time array
t = np.linspace(0, total_t,Number_of_time_intervals)

# Initialize position arrays
true_x = np.zeros_like(t)
true_y = np.zeros_like(t)


for i, ti in enumerate(t):
    if ti < transition_t_1:
        # Circular motion
        theta = omega_A * ti 
        true_x[i] = r_A * -np.cos(theta + pi/2)
        true_y[i] = r_A * np.sin(theta + pi/2)
    #else:
    elif ti < transition_t_2:  
        
        # Linear motion (transition from last circular position)
        true_x[i] = true_x[i-1] + linear_velocity *np.cos(theta_1)* (t[i] - t[i-1])
        true_y[i] = true_y[i-1] - linear_velocity*np.sin(theta_1) *(t[i] - t[i-1]) # Keep y constant
  
    elif ti < transition_t_3:
        # Circular motion 2
        theta = omega_B * ti
        true_x[i] = 100 + r_B * np.cos(theta)
        true_y[i] = r_B * -np.sin(theta)
    
    elif ti < transition_t_4:
        # Linear motion (transition from last circular position)
        true_x[i] = true_x[i-1] - linear_velocity *np.cos(theta_1)* (t[i] - t[i-1])
        true_y[i] = true_y[i-1] - linear_velocity*np.sin(theta_1) *(t[i] - t[i-1]) #

    elif ti < transition_t_5:
        # Circular motion
        theta = omega_A * ti + 2*theta_1
        true_x[i] = r_A * np.cos(theta + pi/2)
        true_y[i] = -r_A * np.sin(theta + pi/2)
    
    
#Fixes the start point as the end point
true_x[-1] = true_x[0]
true_y[-1] = true_y[0]


#Plot the motion
plt.plot(true_x, true_y, label="Motion Path")
plt.scatter(true_x[0], true_y[0], color="green", label="Start")
plt.scatter(true_x[-1], true_y[-1], color="black", label="End")
plt.scatter(transition_points_x, transition_points_y, color='red', s=20)   #transition points
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Transition from Circular to Linear Motion")
plt.legend()
plt.axis("equal")  # Keep aspect ratio consistent
plt.grid()
plt.show()



for i in range(num_steps):
    
    
    true_positions.append((true_x, true_y))
    
    # Noisy measurements
    z = np.array([[true_x + np.random.normal(0, 0.5)],
                  [true_y + np.random.normal(0, 0.5)]])
    measurements.append(z)
    
    # Kalman filter update
    kf.predict()
    kf.update(z)
    
    # Plot results

    plt.scatter(true_x, true_y, color='blue', s=10)  # True position
    plt.scatter(kf.x[0, 0], kf.x[1, 0], color='red', s=10)  # Estimated position


plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Kalman Filter - Circular Motion Tracking")
plt.legend(["True Position", "Estimated Position"])
plt.axis("equal")
plt.show()
