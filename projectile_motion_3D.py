__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from rk4 import Rk4
from matplotlib import pyplot as plt

R_EARTH = 6371e3

def calc_deg(deg, mins, secs):
    return deg + mins/60 + secs/3600

def calc_r_vec(lambd, longitude, z, r = R_EARTH): #    Arguments in radians. z = height of object in respect to the ground.
    return np.array([ r * (np.pi / 2 - lambd), r * np.cos(lambd) * longitude, r + z])

#   Azimuthal equidistant projection
#   Wiki: https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
#       Calculation of the azimuth.
def calc_th(coord1, coord2): # theta is 0 on the pos x-axis and pi/2 on the pos y-axis in the defined coordinate system
    return np.pi - np.arctan2(np.cos(coord2[0])*np.sin(coord2[1]-coord1[1]),
                     (np.cos(coord1[0]) * np.sin(coord2[0]) - np.sin(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1])))
#       Calculation of arc length using the formula of the great circle distance.
def calc_circ_dist(coord1, coord2, r = R_EARTH):
    return r * np.arccos(np.sin(coord1[0]) * np.sin(coord2[0]) + np.cos(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1]))

def interpolate_xl( x0, x1, y0, y1):
    return (x0 - y0 * x1 / y1) / (- y0 / y1 + 1)

LAMBD_CREPY = np.deg2rad(calc_deg(49, 36, 18))
PHI_CREPY = np.deg2rad(calc_deg(3,30,53))
LAMBD_PARIS = np.deg2rad(calc_deg(48, 51, 24))
PHI_PARIS = np.deg2rad(calc_deg(2,21,3))
CREPY_R_VEC = calc_r_vec(LAMBD_CREPY, PHI_CREPY)
PARIS_R_VEC = calc_r_vec(LAMBD_PARIS, PHI_PARIS)

print(calc_circ_dist([-np.pi/2, 0], [np.pi/2, 0]))
print(calc_circ_dist([LAMBD_CREPY, PHI_CREPY], [LAMBD_PARIS, PHI_PARIS]))

class Projectile_3D:
    def __init__(self, lat, long, v0, th0, alph0, m = 106, w = 7.29e-5, r = R_EARTH, name = None):  # Lambda, phi and theta angles are in degrees.
        self.name = name
        #   Initial values
        self.lambd0 = np.radians(lat)   #   LATITUDE
        self.phi0 = np.radians(long)    #   LONGITUDE
        self.th0 = np.radians(th0)      #   SHOOTING DIRECTION
        self.alph0 = np.radians(alph0)  #   SHOOTING ANGLE
        self.r0_vec = calc_r_vec(self.lambd0, self.phi0, 0, r=r)
        self.v0 = v0
        self.m = m
        self.w = w
        self.r = r

        #   Updating variables
        self.r_vec = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.alph = self.alph0
        self.v = self.v0
        self.v_vec = self.v * np.array([np.cos(self.phi) * np.cos(self.th), np.sin(self.phi) * np.cos(self.th), np.sin(self.alph)])
        self.w_vec = np.array([-self.w * np.cos(self.lambd0), 0, self.w * np.sin(self.lambd0)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wr_cross = np.cross(self.w_vec, self.r_vec)
        self.t = 0

    def __str__(self):
        return self.name

    def reset(self):
        self.r_vec = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.v = self.v0
        self.v_vec = self.v0 * np.array([np.cos(self.phi0) * np.cos(self.th0), np.sin(self.phi0) * np.cos(self.th0), np.sin(self.alph0)])
        self.w_vec = np.array([-self.w * np.cos(self.lambd0), 0, self.w * np.sin(self.lambd0)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wr_cross = np.cross(self.w_vec, self.r_vec)
        self.t = 0

    def update_v(self):
        self.v = np.sqrt(np.dot(self.v_vec, self.v_vec))

    def update_lambd(self):
        self.lambd = np.pi/2 - self.r_vec[0]/self.r

    def update_phi(self):
        self.phi = self.r_vec[1]/self.r

    def update_th(self):
        self.th = np.arctan2(self.v_vec[1], self.v_vec[0])

    def update_alph(self):
        self.alph = np.arctan2(self.v_vec[2], np.sqrt(np.dot(self.v_vec[:2], self.v_vec[:2])))

    def set_alph0(self, alph_new):
        self.alph0 = np.radians(alph_new)
        self.reset()

class Motion_3D:    #   VIRTUAL CLASS - USE INHERITED CLASSES INSTEAD
    def __init__(self, projectile, dt, name=None):
        self.name = name
        self.projectile = projectile
        self.dt = dt

        #   Useful details about the motion.
        self.alph0 = self.projectile.alph0
        self.xy_l = np.zeros(2)
        self.z_max = 0
        self.tl = 0

        #   Arrays denoting the position of the projectile.
        self.r_vec_arr = np.array([self.projectile.r_vec])

    def __str__(self):
        return self.name

    #   Second order DEs for a projectile.
    #   Needed for Rk4-algorithm.
    def vx(self, t, x):
        return self.projectile.v_vec[0]

    def vy(self, t, y):
        return self.projectile.v_vec[1]

    def vz(self, t, y):
        return self.projectile.v_vec[2]

    def ax(self, t, vx):
        return 0

    def ay(self, t, vy):
        return 0

    def az(self, t, vy):
        return 0

    def calculate_trajectory(self): #   Uses rk4-algorithm.
        #   Initializing objects of 4th order Runge-Kutta.
        Rk4_vx = Rk4(self.projectile.t0, self.projectile.r_vec[0], self.dt, self.vx)
        Rk4_vy = Rk4(self.projectile.t0, self.projectile.r_vec[1], self.dt, self.vy)
        Rk4_vz = Rk4(self.projectile.t0, self.projectile.r_vec[2], self.dt, self.vy)
        Rk4_ax = Rk4(self.projectile.t0, self.projectile.v_vec[0], self.dt, self.ax)
        Rk4_ay = Rk4(self.projectile.t0, self.projectile.v_vec[1], self.dt, self.ay)
        Rk4_az = Rk4(self.projectile.t0, self.projectile.v_vec[2], self.dt, self.az)

        while(self.projectile.r_vec[2] >= 0):  #    While the projectile is above ground.
            self.r_vec_arr = np.vstack([self.r_vec_arr, self.projectile.r_vec])

            Rk4_vx.rk4()
            Rk4_ax.rk4()
            Rk4_vy.rk4()
            Rk4_ay.rk4()
            Rk4_vz.rk4()
            Rk4_az.rk4()

            if (self.projectile.r_vec[2] < Rk4_vz.yi):
                self.z_max = Rk4_vz.yi

            self.projectile.t += self.dt
            self.projectile.r_vec = np.array([Rk4_vx.yi, Rk4_vy.yi, Rk4_vz.yi])
            self.projectile.v_vec = np.array([Rk4_ax.yi, Rk4_ay.yi, Rk4_az.yi])
            self.projectile.update_v()
            self.projectile.update_lambd()
            self.projectile.update_phi()
            self.projectile.update_th()
            self.projectile.update_alph()

        #   Interpolation of landing point.

        x_l = interpolate_xl(self.r_vec_arr[-1][0], self.projectile.r_vec[0], self.r_vec_arr[-1][2], self.projectile.r_vec[2])
        y_l = interpolate_xl(self.r_vec_arr[-1][1], self.projectile.r_vec[1], self.r_vec_arr[-1][2],self.projectile.r_vec[2])

        self.xy_l = [x_l, y_l]
        self.tl = self.projectile.t - self.dt / 2
        self.projectile.r_vec = [x_l, y_l, self.projectile.r]
        self.r_vec_arr = np.vstack([self.r_vec_arr, self.projectile.r_vec])
        self.projectile.reset()

class Motion_3D_drag(Motion_3D):
    #  Assumes uniform air density everywhere.
    def __init__(self, projectile, dt, name=None, g=9.81, b = 2e-3):  #   Constants as given in compulsory exercise 1 description/appendix.
        super().__init__(projectile, dt, name)
        self.g = g
        self.bpm = b/self.projectile.m

    #   Second-order DEs for a projectile with drag.
    #   Needed for Rk4-algorithm
    def ax(self, t, vx):
        return -self.bpm*self.projectile.v*vx

    def ay(self, t, vy):
        return -self.bpm*self.projectile.v*vy

    def az(self, t, vz):
        return -self.g-self.bpm*self.projectile.v*vz

#   Air density correction model 2: Adiabatic approximation.
class Motion_2D_drag_adiabatic(Motion_3D_drag):
    def __init__(self, projectile, dt, name = None, g=9.81, b = 2e-3, a=6.5e-3, alpha=2.5, T0=288.2):   #   Constants as given in compulsory exercise 1 description/appendix. We assume that T0 = 15 C.
        super().__init__(projectile, dt, g, b)
        self.name = name
        self.a = a
        self.alpha = alpha
        self.T0 = T0

    def rhofrac_adiabatic(self):
        #   rho/rho0 as given in the appendix.
        return (1 - self.a / self.T0 * (self.projectile.r_vec[2]-self.projectile.r)) ** self.alpha

    def ax(self, t, vx):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vx

    def ay(self, t, vy):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vy

    def az(self, t, vz):
        return -self.g - self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vz