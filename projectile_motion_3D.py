__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from rk4 import Rk4
from matplotlib import pyplot as plt

R_EARTH = 6371e3

def calc_deg(deg, mins, secs):
    return deg + mins/60 + secs/3600

def calc_r_vec(lambd, longitude, r = R_EARTH):
    return r * np.array([(np.pi / 2 - lambd) / (2 * np.pi), np.cos(lambd) * longitude / (2 * np.pi), 1])

def calc_th(coord1, coord2): # theta is 0 on the x-axis in the defined coordinate system
    return np.pi - np.arctan2(np.cos(coord2[0])*np.sin(coord2[1]-coord1[1]),
                     (np.cos(coord1[0]) * np.sin(coord2[0]) - np.sin(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1])))

def calc_circ_dist(coord1, coord2, r = R_EARTH):
    return r * np.arccos(np.sin(coord1[0]) * np.sin(coord2[0]) + np.cos(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1])) / (2* np.pi)

LAMBD_CREPY = np.deg2rad(calc_deg(49, 36, 18))
PHI_CREPY = np.deg2rad(calc_deg(3,30,53))
LAMBD_PARIS = np.deg2rad(calc_deg(48, 51, 24))
PHI_PARIS = np.deg2rad(calc_deg(2,21,3))
CREPY_R_VEC = calc_r_vec(LAMBD_CREPY, PHI_CREPY)
PARIS_R_VEC = calc_r_vec(LAMBD_PARIS, PHI_PARIS)

print(calc_circ_dist([0, 0], [0, np.pi]))
print(calc_circ_dist([LAMBD_CREPY, PHI_CREPY], [LAMBD_PARIS, PHI_PARIS]))

class Projectile_3D:
    def __init__(self, lat, long, v0, th0, alph0, w = 7.29e-5, name = None):  # Lambda, phi and theta angles are in degrees.
        self.name = name
        #   Initial values
        self.lambd0 = np.radians(lat)
        self.phi0 = np.radians(long)
        self.th0 = np.radians(th0)
        self.r0_vec = calc_r_vec(self.lambd0, phi0)
        self.v0 = v0
        self.w = w

        #   Updating variables
        self.r_vec = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.v = self.v0
        self.v_vec = self.v0 * np.array([np.cos(self.phi0) * np.cos(self.th0), np.sin(self.phi0) * np.cos(self.th0), np.sin(self.th0)])
        self.w_vec = np.array([-self.w * np.cos(self.lambd0), 0, self.w * np.sin(self.lambd0)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wr_cross = np.cross(self.w_vec, self.r_vec)
        self.t = 0

    def __str__(self):
        return self.name

    def reset(self):
        self.r = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.v = self.v0
        self.v_vec = self.v0 * np.array([np.cos(self.phi0) * np.cos(self.th0), np.sin(self.phi0) * np.cos(self.th0), np.sin(self.th0)])
        self.w_vec = np.array([-self.w * np.cos(self.lambd0), 0, self.w * np.sin(self.lambd0)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wr_cross = np.cross(self.w_vec, self.r_vec)
        self.t = 0

    def update_th(self):
        self.th = np.arctan(self.v_vec[1]/self.v_vec[0])

    def update_v(self):
        self.v = np.sqrt(self.v_vec[0]**2+self.v_vec[1]**2)

    def set_th0(self, th_new):
        self.th0 = np.radians(th_new)
        self.reset()