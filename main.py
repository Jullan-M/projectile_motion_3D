__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import utilities as ut
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from projectile_motion_3D import Projectile_3D, Motion_3D_drag_adiabatic, Motion_3D_drag_adiabatic_coriolis

R_EARTH = 6371e3
COORD_CREPY = [ut.calc_deg(49, 36, 18), ut.calc_deg(3,30,53)]
COORD_PARIS = [ut.calc_deg(48, 51, 24), ut.calc_deg(2,21,3)]

if (__name__ == "__main__"):
    th1 = ut.calc_th(COORD_CREPY, COORD_PARIS)
    proj1 = Projectile_3D(COORD_CREPY, 1640, th1 + 1.78, 34.6897)
    mo1 = Motion_3D_drag_adiabatic(proj1, 0.01)
    mo1.calculate_trajectory()

    print("Crepy coords:\t\t", COORD_CREPY, "deg")
    print("Paris coords:\t\t", COORD_PARIS, "deg")
    print("Projectile coords:\t", ut.r_vec_to_coord(mo1.r_vec_arr[-1]), "deg")
    print("Crepy - Paris azimuth:\t", th1, "deg")
    print("Crepy - Paris distance:\t", R_EARTH * ut.calc_circ_dist(np.radians(COORD_CREPY), np.radians(COORD_PARIS)),
          "meters")
    print("Projectile distance:\t", mo1.distance, "meters")
    print("Projectile angle:\t", 360 + np.rad2deg(proj1.th), "\tstart:\t", np.rad2deg(proj1.th0))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')

    u = np.radians(np.linspace(COORD_CREPY[1], COORD_PARIS[1], 100))
    v = np.radians(np.linspace(COORD_CREPY[0], COORD_PARIS[0], 100))

    mo1_line = np.apply_along_axis(ut.r_vec_to_cartesian, 1, mo1.r_vec_arr)
    crepy_paris_line = R_EARTH * np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    earth = R_EARTH * np.array([np.outer(np.cos(u), np.cos(v)),
                                np.outer(np.sin(u), np.cos(v)), np.outer(np.ones(np.size(u)),
                                                                         np.sin(v))])
    crepy = R_EARTH * ut.coord_to_cartesian(COORD_CREPY)
    paris = R_EARTH * ut.coord_to_cartesian(COORD_PARIS)
    crash = ut.r_vec_to_cartesian(proj1.r_vec)

    ax.scatter(crepy[0], crepy[1], crepy[2], c="r", marker="^", label=r"Crépy")
    ax.scatter(paris[0], paris[1], paris[2], c="g", marker="^", label=r"Paris")
    ax.scatter(crash[0], crash[1], crash[2], c="m", marker="x", label=r"LZ")
    ax.plot(crepy_paris_line[0], crepy_paris_line[1], crepy_paris_line[2])
    ax.plot(mo1_line[:, 0], mo1_line[:, 1], mo1_line[:, 2], label=r"Proj trajectory")
    ax.plot_surface(earth[0], earth[1], earth[2], color='k', alpha=0.1)
    plt.legend()
    ax.view_init(azim=-67.5, elev=45)
    # plt.savefig("trajectory.pdf")
    plt.show()