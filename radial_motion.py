__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan

import numpy as np
from rk4 import Rk4
from matplotlib import pyplot as plt

class ProjectileRadial:
    def __init__(self, r0, th0, T, m):  #  Units are AU, radians, yrs, kg respectively
        #   Initial
        self.r0 = r0
        self.th0 = th0
        self.T = T
        self.v0 = 2*np.pi*self.r0/T
        self.m = m

        #   Updating variables
        self.r = self.r0
        self.th = self.th0
        self.v = self.v0
        self.T = 0.5*self.m*self.v0**2
        self.V = 0  # Potential will be determined when an ProjectileRadial object is used in a MotionRadial object.
        self.E = self.T + self.V


    def __str__(self):
        return "Radius: " + self.r + "\tTheta: " + self.th + "\tTime: " + self.t

    def reset(self):
        self.r = self.r0
        self.th = self.th0

def potential(r):
    return 

class MotionRadial:
    def __init__(self, projectile, potential , dt):
        self.projectile = projectile
        self.dt = dt
        self.t = 0
