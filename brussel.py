__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import utilities as ut
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from projectile_motion_3D import Projectile_3D, Motion_3D_drag_adiabatic, Motion_3D_drag_adiabatic_coriolis
import timeit
from main import Plotter

R_EARTH = 6371e3
COORD_CREPY = [ut.calc_deg(49, 36, 18), ut.calc_deg(3,30,53)]
COORD_BRUSSEL = [50.847945, 4.350422]

if (__name__ == "__main__"):
    azimuth = ut.calc_azimuth(COORD_CREPY, COORD_BRUSSEL)
    th1 = ut.calc_th(COORD_CREPY, COORD_BRUSSEL)
    add = 0.4166
    alph0 = 46.0787
    proj1 = Projectile_3D(COORD_CREPY, 1640, th1 + add, alph0)
    mo1 = Motion_3D_drag_adiabatic_coriolis(proj1, 0.01)
    mo1.calculate_trajectory()
    plot1 = Plotter(mo1)

    crepy_brussel_dist = R_EARTH * ut.calc_circ_dist(np.radians(COORD_CREPY), np.radians(COORD_BRUSSEL))
    brussel_mo1_dist = proj1.r * ut.calc_circ_dist(np.radians(COORD_BRUSSEL),
                                                 np.radians(ut.r_vec_to_coord(mo1.r_vec_arr[-1])))

    print("Crepy coords:\t\t", COORD_CREPY, "deg")
    print("Brussel coords:\t\t", COORD_BRUSSEL, "deg")
    print("Crepy - Brussel azimuth:\t", azimuth, "deg")
    print("Shooting direction without coriolis:\t", th1, "deg")
    print("Shooting direction:\t", th1 + add, "deg")
    print("Shooting angle:\t", alph0, "deg")
    print("Crepy - Brussel distance:\t", crepy_brussel_dist, "meters")

    print("\t=== CORIOLIS ON ===")
    print("Max height:\t", mo1.z_max, "m")
    print("Time:\t", mo1.tl, "s")
    print("Coords:\t", ut.r_vec_to_coord(mo1.r_vec_arr[-1]), "deg")
    print("Distance:\t", mo1.distance, "m")
    print("Angle:\t", 360 + np.rad2deg(mo1.th), "deg\tstart:\t", np.rad2deg(proj1.th0), "deg")
    print("Distance from Brussel:\t", brussel_mo1_dist, "m")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')

    #   Marking of cities
    crepy = R_EARTH * ut.coord_to_cartesian(COORD_CREPY)
    paris = R_EARTH * ut.coord_to_cartesian(COORD_BRUSSEL)
    ax.scatter(crepy[0], crepy[1], crepy[2], c="r", marker="^", label=r"Crépy")
    ax.scatter(paris[0], paris[1], paris[2], c="g", marker="^", label=r"Brussel")

    #   Marking of crash sites
    ax.scatter(plot1.crash[0], plot1.crash[1], plot1.crash[2], c="k", marker="x")

    #   Plotting of chords, trajectory curves and surfaces
    u = np.radians(np.linspace(COORD_CREPY[1], COORD_BRUSSEL[1], 100))
    v = np.radians(np.linspace(COORD_CREPY[0], COORD_BRUSSEL[0], 100))
    crepy_brussel_line = R_EARTH * np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    earth = R_EARTH * np.array([np.outer(np.cos(u), np.cos(v)),
                                np.outer(np.sin(u), np.cos(v)), np.outer(np.ones(np.size(u)),
                                                                         np.sin(v))])
    ax.plot(crepy_brussel_line[0], crepy_brussel_line[1], crepy_brussel_line[2], label=r"Surface", alpha=0.5)
    ax.plot(plot1.line_3d[0], plot1.line_3d[1], plot1.line_3d[2], c="k", label=r"With coriolis", linewidth=0.8)
    ax.plot_surface(earth[0], earth[1], earth[2], color='b', alpha=0.1)

    plt.legend()
    ax.view_init(azim=45, elev=45)
    plt.savefig("3D_trajectory_brussel.pdf")
    plt.show()

    plt.figure()
    plt.plot(plot1.dist, plot1.height, label=r"With coriolis", color="k")
    plt.plot(0, 0, "r^", label=r"Crépy")
    plt.plot(crepy_brussel_dist, 0, "g^", label=r"Paris")
    plt.xlabel(r"Distance", fontsize=16)
    plt.ylabel(r"Height, $z$", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("2D_trajectory_brussel.pdf")
    plt.show()

    plt.figure()
    plt.plot(plot1.long, plot1.lat, label=r"With coriolis", color="k")
    plt.plot(plot1.long[-1], plot1.lat[-1], "kx")
    plt.plot(COORD_CREPY[1], COORD_CREPY[0], 'r^', label=r"Crépy")
    plt.plot(COORD_BRUSSEL[1], COORD_BRUSSEL[0], 'g^', label=r"Brussel")
    #plt.xlim(left=4.3504, right=4.3505)#plt.xlim(left=2.350, right=2.351)    #   (left=2.3508, right=2.3509)
    #plt.ylim(bottom=50.8479, top=50.848)#plt.ylim(bottom=48.856, top=48.857) #   (bottom=48.8566, top=48.8567)
    plt.xlabel(r"Longitude", fontsize=16)
    plt.ylabel(r"Latitude", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("lat_vs_long_trajectory_brussel.pdf")
    plt.show()
