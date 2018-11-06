__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from rk4 import Rk4
import utilities as ut

class Projectile_3D:
    def __init__(self, coord, v0, th0, alph0, m = 106, w = 7.29e-5, r = 6371e3, name = None):  # Latitude, longitude and theta arguments are all in degrees.
        self.name = name
        #   Initial values
        self.lambd0 = np.radians(coord[0])   #   LATITUDE
        self.phi0 = np.radians(coord[1])    #   LONGITUDE
        self.th0 = np.radians(th0)      #   SHOOTING DIRECTION
        self.alph0 = np.radians(alph0)  #   SHOOTING ANGLE
        self.r0_vec = np.array([ r * (np.pi / 2 - self.lambd0), r * np.cos(self.lambd0) * self.phi0, r ])
        self.v0 = v0

        #   Constants
        self.m = m
        self.w = w
        self.r = r
        self.t0 = 0

        #   Updating variables
        self.r_vec = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.alph = self.alph0
        self.v = self.v0
        self.v_vec = self.v * np.array([np.cos(self.alph) * np.cos(self.th), np.cos(self.alph) * np.sin(self.th), np.sin(self.alph)])
        self.w_vec = self.w * np.array([- np.cos(self.lambd), 0, np.sin(self.lambd)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wwr_cross =  np.cross(self.w_vec, np.cross(self.w_vec, self.r_vec))
        self.t = self.t0

    def __str__(self):
        return self.name

    def reset(self):
        self.r_vec = np.copy(self.r0_vec)
        self.lambd = self.lambd0
        self.phi = self.phi0
        self.th = self.th0
        self.v = self.v0
        self.v_vec = self.v0 * np.array([np.cos(self.alph0) * np.cos(self.th0), np.cos(self.alph0) * np.sin(self.th0), np.sin(self.alph0)])
        self.w_vec = np.array([-self.w * np.cos(self.lambd), 0, self.w * np.sin(self.lambd)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wwr_cross = np.cross(self.w_vec, np.cross(self.w_vec, self.r_vec))
        self.t = 0

    def update_v(self):
        self.v = np.sqrt(np.dot(self.v_vec, self.v_vec))

    def update_lambd(self):
        self.lambd = np.pi/2 - self.r_vec[0]/self.r

    def update_phi(self):   #   NOTE: lambd has to be updated BEFORE phi.
        self.phi = self.r_vec[1]/(np.cos(self.lambd)*self.r)

    def update_th(self):
        self.th = np.arctan2(self.v_vec[1], self.v_vec[0])

    def update_alph(self):
        self.alph = np.arctan2(self.v_vec[2], np.sqrt(np.dot(self.v_vec[:2], self.v_vec[:2])))

    def update_w_vec(self): #   NOTE: lambd has to be updated BEFORE w_vec.
        self.w_vec = self.w * np.array([- np.cos(self.lambd), 0, np.sin(self.lambd)])

    #   Cross-products - w_vec has to be updated before these.
    def update_wv_cross(self):
        self.wv_cross = np.cross(self.w_vec, self.v_vec)

    def update_wwr_cross(self):
        self.wwr_cross = np.cross(self.w_vec, np.cross(self.w_vec, self.r_vec))

    def update_all(self):
        self.update_v()
        self.update_lambd()
        self.update_phi()
        self.update_th()
        self.update_alph()
        self.update_w_vec()
        self.update_wv_cross()
        self.update_wwr_cross()

    def set_alph0(self, alph_new):
        self.alph0 = np.radians(alph_new)

    def set_th0(self, th_new):
        self.th0 = np.radians(th_new)

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
        self.distance = 0
        self.th = self.projectile.th

        #   Arrays denoting the position of the projectile.
        self.r_vec_arr = np.empty(3)

    def __str__(self):
        return self.name

    def calc_distance(self):
        self.distance = self.projectile.r * ut.calc_circ_dist([self.projectile.lambd0, self.projectile.phi0],
                                                [self.projectile.lambd, self.projectile.phi])

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

    def az(self, t, vz):
        return 0

    def calculate_trajectory(self): #   Uses rk4-algorithm.
        #   Initializing objects of 4th order Runge-Kutta.
        Rk4_vx = Rk4(self.projectile.t0, self.projectile.r_vec[0], self.dt, self.vx)
        Rk4_vy = Rk4(self.projectile.t0, self.projectile.r_vec[1], self.dt, self.vy)
        Rk4_vz = Rk4(self.projectile.t0, self.projectile.r_vec[2], self.dt, self.vz)
        Rk4_ax = Rk4(self.projectile.t0, self.projectile.v_vec[0], self.dt, self.ax)
        Rk4_ay = Rk4(self.projectile.t0, self.projectile.v_vec[1], self.dt, self.ay)
        Rk4_az = Rk4(self.projectile.t0, self.projectile.v_vec[2], self.dt, self.az)

        while(self.projectile.r_vec[2] >= self.projectile.r):  #    While the projectile is above ground.
            self.r_vec_arr = np.vstack([self.r_vec_arr, self.projectile.r_vec])

            Rk4_vx.rk4()
            Rk4_vy.rk4()
            Rk4_vz.rk4()
            Rk4_ax.rk4()
            Rk4_ay.rk4()
            Rk4_az.rk4()

            #   Updates max reached height of projectile
            #   in relation to ground level.
            if (self.projectile.r_vec[2] < Rk4_vz.yi):
                self.z_max = Rk4_vz.yi - self.projectile.r

            self.projectile.t += self.dt

            self.projectile.r_vec = np.array([Rk4_vx.yi
                                              ,Rk4_vy.yi #- self.projectile.r_vec[1] + self.projectile.r * np.sin(Rk4_vx.yi / self.projectile.r) * self.projectile.phi
                                              ,Rk4_vz.yi])
            self.projectile.v_vec = np.array([Rk4_ax.yi, Rk4_ay.yi, Rk4_az.yi])
            self.projectile.update_all()

        #   Removal of the empty vector.
        self.r_vec_arr = self.r_vec_arr[1:]

        #   Interpolation of landing point.
        x_l = ut.interpolate_xl(self.r_vec_arr[-1][0], self.projectile.r_vec[0], self.r_vec_arr[-1][2] - self.projectile.r, self.projectile.r_vec[2] - self.projectile.r)
        y_l = ut.interpolate_xl(self.r_vec_arr[-1][1], self.projectile.r_vec[1], self.r_vec_arr[-1][2] - self.projectile.r, self.projectile.r_vec[2] - self.projectile.r)
        self.xy_l = [x_l, y_l]
        self.tl = self.projectile.t - self.dt / 2

        #   Appending the last r_vec
        self.projectile.r_vec = [x_l, y_l, self.projectile.r]
        self.projectile.update_all()
        self.r_vec_arr = np.vstack([self.r_vec_arr, self.projectile.r_vec])

        #   Updating relevant values of the motion
        self.th = self.projectile.th
        self.calc_distance()

class Motion_3D_drag(Motion_3D):
    #   Assumes uniform air density everywhere.
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
class Motion_3D_drag_adiabatic(Motion_3D_drag):
    def __init__(self, projectile, dt, name = None, g=9.81, b = 2e-3, a=6.5e-3, alpha=2.5, T0=288.2):   #   Constants as given in compulsory exercise 1 description/appendix. We assume that T0 = 15 C.
        super().__init__(projectile, dt, name, g, b)
        self.a = a
        self.alpha = alpha
        self.T0 = T0

    def rhofrac_adiabatic(self):
        #   rho/rho0 as given in the appendix.
        value = (1 - self.a / self.T0 * (self.projectile.r_vec[2]-self.projectile.r))
        overZero = value > 0
        return (overZero * value) ** self.alpha

    def ax(self, t, vx):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vx

    def ay(self, t, vy):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vy

    def az(self, t, vz):
        return -self.g - self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vz

class Motion_3D_drag_adiabatic_coriolis(Motion_3D_drag_adiabatic):
    def __init__(self, projectile, dt, name = None, g=9.81, b = 2e-3, a=6.5e-3, alpha=2.5, T0=288.2):   #   Constants as given in compulsory exercise 1 description/appendix. We assume that T0 = 15 C.
        super().__init__(projectile, dt, name, g, b, a, alpha, T0)

    def ax(self, t, vx):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vx - 2 * self.projectile.wv_cross[0] #- self.projectile.wwr_cross[0]

    def ay(self, t, vy):
        return -self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vy - 2 * self.projectile.wv_cross[1] #- self.projectile.wwr_cross[1]

    def az(self, t, vz):
        return -self.g - self.bpm * self.rhofrac_adiabatic() * self.projectile.v*vz - 2 * self.projectile.wv_cross[2] #- self.projectile.wwr_cross[2]

