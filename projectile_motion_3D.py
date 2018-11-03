__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
from rk4 import Rk4
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


R_EARTH = 6371e3

def calc_deg(deg, mins, secs):
    return deg + mins/60 + secs/3600

def r_vec_to_coord(r_vec):
    lambd = np.pi / 2 - r_vec[0] / r_vec[2]
    phi = r_vec[1] / (np.cos(lambd) * r_vec[2])
    return np.rad2deg(np.array([lambd, phi/(np.cos(lambd)*r_vec[2])]))

def r_vec_to_cartesian(r_vec):
    lambd = np.pi/2-r_vec[0]/r_vec[2]
    phi = r_vec[1]/(np.cos(lambd)*r_vec[2])
    return r_vec[2] * np.array([np.cos(lambd) * np.cos(phi),
                                np.cos(lambd) * np.sin(phi),
                                np.sin(lambd)])

def coord_to_cartesian(coord, r = R_EARTH):
    coord_rad = np.radians(coord)
    return r * np.array([np.cos(coord_rad[0]) * np.cos(coord_rad[1]),
                        np.cos(coord_rad[0]) * np.sin(coord_rad[1]),
                        np.sin(coord_rad[0])])


#   Azimuthal equidistant projection
#   Wiki: https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
#       Calculation of the azimuth.
def calc_th(coord1_deg, coord2_deg): # theta is 0 on the pos x-axis and pi/2 on the pos y-axis in the defined coordinate system
    coord1 = np.radians(coord1_deg)
    coord2 = np.radians(coord2_deg)
    return np.rad2deg(np.pi - np.arctan2(np.cos(coord2[0])*np.sin(coord2[1]-coord1[1]),
                     (np.cos(coord1[0]) * np.sin(coord2[0]) - np.sin(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1]))))
#       Calculation of arc length using the formula of the great circle distance.
def calc_circ_dist(coord1, coord2):
    return np.arccos(np.sin(coord1[0]) * np.sin(coord2[0]) + np.cos(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1]))

def interpolate_xl( x0, x1, y0, y1):
    return (x0 - y0 * x1 / y1) / (- y0 / y1 + 1)

COORD_CREPY = [calc_deg(49, 36, 18), calc_deg(3,30,53)]
COORD_PARIS = [calc_deg(48, 51, 24), calc_deg(2,21,3)]

class Projectile_3D:
    def __init__(self, coord, v0, th0, alph0, m = 106, w = 7.29e-5, r = R_EARTH, name = None):  # Latitude, longitude and theta arguments are all in degrees.
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
        self.w_vec = np.array([-self.w * np.cos(self.lambd0), 0, self.w * np.sin(self.lambd0)])

        self.wv_cross = np.cross(self.w_vec, self.v_vec)
        self.wr_cross = np.cross(self.w_vec, self.r_vec)
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

    def update_wv_cross(self):
        self.wv_cross = np.cross(self.w_vec, self.v_vec)

    def update_wr_cross(self):
        self.wr_cross = np.cross(self.w_vec, self.r_vec)

    def update_all(self):
        self.update_v()
        self.update_lambd()
        self.update_phi()
        self.update_th()
        self.update_alph()
        self.update_wv_cross()
        self.update_wr_cross()

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
        self.distance = 0

        #   Arrays denoting the position of the projectile.
        self.r_vec_arr = np.empty(3)

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
            Rk4_ax.rk4()
            Rk4_vy.rk4()
            Rk4_ay.rk4()
            Rk4_vz.rk4()
            Rk4_az.rk4()

            if (self.projectile.r_vec[2] < Rk4_vz.yi):
                self.z_max = Rk4_vz.yi -self.projectile.r

            self.projectile.t += self.dt
            self.projectile.r_vec = np.array([Rk4_vx.yi, Rk4_vy.yi, Rk4_vz.yi])
            self.projectile.v_vec = np.array([Rk4_ax.yi, Rk4_ay.yi, Rk4_az.yi])
            self.projectile.update_all()
        #   Removes the empty vector.
        self.r_vec_arr = self.r_vec_arr[1:]

        #   Interpolation of landing point.
        x_l = interpolate_xl(self.r_vec_arr[-1][0], self.projectile.r_vec[0], self.r_vec_arr[-1][2] - self.projectile.r, self.projectile.r_vec[2] - self.projectile.r)
        y_l = interpolate_xl(self.r_vec_arr[-1][1], self.projectile.r_vec[1], self.r_vec_arr[-1][2] - self.projectile.r, self.projectile.r_vec[2] - self.projectile.r)
        self.xy_l = [x_l, y_l]
        self.tl = self.projectile.t - self.dt / 2

        self.projectile.r_vec = [x_l, y_l, self.projectile.r]
        self.projectile.update_all()
        self.distance = self.projectile.r * calc_circ_dist([self.projectile.lambd0, self.projectile.phi0],
                                                [self.projectile.lambd, self.projectile.phi])
        self.r_vec_arr = np.vstack([self.r_vec_arr, self.projectile.r_vec])

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
        self.name = name
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

if (__name__ == "__main__"):
    th1 = calc_th(COORD_CREPY, COORD_PARIS)
    proj1 = Projectile_3D(COORD_CREPY, 1640, th1, 33)
    mo1 = Motion_3D_drag_adiabatic(proj1, 0.01)
    mo1.calculate_trajectory()
    mo1_line = np.apply_along_axis(r_vec_to_cartesian, 1, mo1.r_vec_arr)
    print(COORD_PARIS)
    print(r_vec_to_coord(mo1.r_vec_arr[-1]))
    '''
    ========================
    3D surface (solid color)
    ========================
    
    Demonstrates a very basic plot of a 3D surface using a solid color.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')
    """
    u = np.linspace(0, np.pi/4, 100)
    v = np.linspace(0, np.pi/4, 100)
    earth = R_EARTH * np.array([np.outer(np.cos(u), np.sin(v)),
                                np.outer(np.sin(u), np.sin(v)), np.outer(np.ones(np.size(u)),
                                np.cos(v))])
    
    
    # Plot the surface
    ax.plot_wireframe(earth[0], earth[1], earth[2], color='b')
    """
    crepy = coord_to_cartesian(COORD_CREPY)
    paris = coord_to_cartesian(COORD_PARIS)
    crepy_paris = np.radians(np.array([np.linspace(COORD_CREPY[0], COORD_PARIS[0], 100), np.linspace(COORD_CREPY[1], COORD_PARIS[1], 100)]))
    crepy_paris_line = R_EARTH * np.array([np.cos(crepy_paris[0])*np.cos(crepy_paris[1]), np.cos(crepy_paris[0])*np.sin(crepy_paris[1]), np.sin(crepy_paris[0])])
    print(mo1_line[0])
    print(crepy)
    print("\n", mo1_line[-1], sep="")
    print(paris)
    ax.scatter(crepy[0], crepy[1], crepy[2], c="r", marker="^")
    ax.scatter(paris[0], paris[1], paris[2], c="g", marker="^")
    ax.plot(crepy_paris_line[0], crepy_paris_line[1], crepy_paris_line[2])
    ax.plot(mo1_line[:,0], mo1_line[:,1], mo1_line[:,2])

    ax.view_init(azim=45, elev=45)

    plt.show()
