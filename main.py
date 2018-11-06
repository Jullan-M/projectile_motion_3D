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
    proj1 = Projectile_3D(COORD_CREPY, 1640, th1 #+ 1.78
                          , 34.6897)
    mo1 = Motion_3D_drag_adiabatic(proj1, 0.01)
    mo2 = Motion_3D_drag_adiabatic_coriolis(proj1, 0.01)
    mo1.calculate_trajectory()
    proj1.reset()
    mo2.calculate_trajectory()
    mo1_mo2_dist = proj1.r * ut.calc_circ_dist(np.radians(ut.r_vec_to_coord(mo1.r_vec_arr[-1])),
                                          np.radians(ut.r_vec_to_coord(mo2.r_vec_arr[-1])))
    paris_mo1_dist = proj1.r * ut.calc_circ_dist(np.radians(COORD_PARIS),
                                          np.radians(ut.r_vec_to_coord(mo1.r_vec_arr[-1])))
    paris_mo2_dist = proj1.r * ut.calc_circ_dist(np.radians(COORD_PARIS),
                                                 np.radians(ut.r_vec_to_coord(mo2.r_vec_arr[-1])))

    print("Crepy coords:\t\t", COORD_CREPY, "deg")
    print("Paris coords:\t\t", COORD_PARIS, "deg")
    print("Crepy - Paris azimuth:\t", th1, "deg")
    print("Crepy - Paris distance:\t", R_EARTH * ut.calc_circ_dist(np.radians(COORD_CREPY), np.radians(COORD_PARIS)),
          "meters")

    print("\t=== CORIOLIS OFF ===")
    print("Max height:\t", mo1.z_max, "m")
    print("Coords:\t", ut.r_vec_to_coord(mo1.r_vec_arr[-1]), "deg")
    print("Distance:\t", mo1.distance, "m")
    print("Angle:\t", 360 + np.rad2deg(mo1.th), "deg\tstart:\t", np.rad2deg(proj1.th0), "deg")
    print("Distance from Paris:\t", paris_mo1_dist, "m")

    print("\t=== CORIOLIS ON ===")
    print("Max height:\t", mo2.z_max, "m\tdelta =", mo2.z_max - mo1.z_max, "m")
    print("Coords:\t", ut.r_vec_to_coord(mo2.r_vec_arr[-1]), "deg")
    print("Distance:\t", mo2.distance, "m\tdelta =" ,mo2.distance-mo1.distance, "m")
    print("Angle:\t", 360 + np.rad2deg(mo2.th), "deg\t\tdelta =", np.rad2deg(mo2.th) - np.rad2deg(mo1.th), "deg")
    print("Distance from Paris:\t", paris_mo2_dist, "m")
    print("\nDistance between LZs:\t", mo1_mo2_dist)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')

    #   Marking of cities
    crepy = R_EARTH * ut.coord_to_cartesian(COORD_CREPY)
    paris = R_EARTH * ut.coord_to_cartesian(COORD_PARIS)
    ax.scatter(crepy[0], crepy[1], crepy[2], c="g", marker="^", label=r"Crépy")
    ax.scatter(paris[0], paris[1], paris[2], c="r", marker="^", label=r"Paris")

    #   Marking of crash sites
    crash1 = ut.r_vec_to_cartesian(mo1.r_vec_arr[-1])
    crash2 = ut.r_vec_to_cartesian(mo2.r_vec_arr[-1])
    ax.scatter(crash1[0], crash1[1], crash1[2], c="k", marker="x")
    ax.scatter(crash2[0], crash2[1], crash2[2], c="m", marker="x")

    #   Plotting of chords, trajectory curves and surfaces
    u = np.radians(np.linspace(COORD_CREPY[1], COORD_PARIS[1], 100))
    v = np.radians(np.linspace(COORD_CREPY[0], COORD_PARIS[0], 100))
    mo1_line = np.apply_along_axis(ut.r_vec_to_cartesian, 1, mo1.r_vec_arr)
    mo2_line = np.apply_along_axis(ut.r_vec_to_cartesian, 1, mo2.r_vec_arr)
    crepy_paris_line = R_EARTH * np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    earth = R_EARTH * np.array([np.outer(np.cos(u), np.cos(v)),
                                np.outer(np.sin(u), np.cos(v)), np.outer(np.ones(np.size(u)),
                                                                         np.sin(v))])
    ax.plot(crepy_paris_line[0], crepy_paris_line[1], crepy_paris_line[2], label=r"Surface", alpha=0.5)
    ax.plot(mo1_line[:, 0], mo1_line[:, 1], mo1_line[:, 2], c="k", label=r"Without coriolis", linewidth=0.8)
    ax.plot(mo2_line[:, 0], mo2_line[:, 1], mo2_line[:, 2], c="m", label=r"With coriolis", linewidth=0.8)
    ax.plot_surface(earth[0], earth[1], earth[2], color='b', alpha=0.1)

    plt.legend()
    ax.view_init(azim=-45, elev=45)
    # plt.savefig("trajectory.pdf")
    plt.show()