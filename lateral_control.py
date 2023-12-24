import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=0.05, damping_constant=0.01):
        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''

        # derive orientation error as the angle of the first path segment to the car orientation
        x = waypoints[0][1] - waypoints[0][0]
        y = waypoints[1][1] - waypoints[1][0]

        dir_vec = np.array([x, y])
        u_dir_vec = dir_vec / np.linalg.norm(dir_vec)
        orientation_error = -np.arcsin(-u_dir_vec[0])

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        car_position = np.array([47.5, 25])
        first_waypoint = np.array([waypoints[0][0], waypoints[0][1]])
        error = np.linalg.norm(first_waypoint - car_position)
        if orientation_error < 0:
            error = -error

        # prevent division by zero by adding as small epsilon
        eps = 0.000001
        steering_angle = orientation_error + np.arctan(error * self.gain_constant / (eps + speed))

        # derive damping
        damping = steering_angle - self.damping_constant * (steering_angle - self.previous_steering_angle)
        self.previous_steering_angle = steering_angle

        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(damping, a_min=-0.4, a_max=0.4)
