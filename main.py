__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np
import utilities as ut
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from projectile_motion_3D import Projectile_3D, Motion_3D_drag_adiabatic, Motion_3D_drag_adiabatic_coriolis
import timeit

R_EARTH = 6371e3
COORD_CREPY = [ut.calc_deg(49, 36, 18), ut.calc_deg(3,30,53)]
COORD_PARIS = [ut.calc_deg(48, 51, 24), ut.calc_deg(2,21,3)]

class Plotter:
    def __init__(self, motion):
        self.motion = motion
        #   Altitude vs. distance plot
        self.dist = self.motion.projectile.r * self.motion.circ_dist_arr
        #   np.sqrt(np.einsum('ij,ij->i', motion.r_vec_arr[:,:2] - motion.r_vec_arr[0,:2], motion.r_vec_arr[:,:2] - motion.r_vec_arr[0,:2])) #    dot product of the 2 first indexes of the r_vecs and then sqrt
        self.height = motion.r_vec_arr[:,2] - self.motion.projectile.r
        #   Latitude vs. longitude plot
        self.lat = np.pi / 2 - motion.r_vec_arr[:, 0] / self.motion.projectile.r
        self.long = np.rad2deg(motion.r_vec_arr[:,1]/(np.cos(self.lat)*self.motion.projectile.r))
        self.lat = np.rad2deg(self.lat)

        #   3D-plot
        self.start = ut.r_vec_to_cartesian(motion.r_vec_arr[0])
        self.crash = ut.r_vec_to_cartesian(motion.r_vec_arr[-1])
        self.line_3d = np.transpose(np.apply_along_axis(ut.r_vec_to_cartesian, 1, motion.r_vec_arr))

if (__name__ == "__main__"):
    azimuth = ut.calc_azimuth(COORD_CREPY, COORD_PARIS)
    th1 = ut.calc_th(COORD_CREPY, COORD_PARIS)
    add = 0
    alph0 = 34.185 #   34.6851 - without curvature corrections    #   max travel distance: 51.3  #   34.185, 68.21956 - best i can do
    proj1 = Projectile_3D(COORD_CREPY, 1640, th1 + add, alph0)
    proj2 = Projectile_3D(COORD_CREPY, 1640, th1 + add, alph0)
    mo1 = Motion_3D_drag_adiabatic(proj1, 0.01)
    mo2 = Motion_3D_drag_adiabatic_coriolis(proj2, 0.01)
    motions = [mo1, mo2]
    start = timeit.default_timer()
    mo1.calculate_trajectory()
    mo2.calculate_trajectory()
    stop = timeit.default_timer()
    print("Time:", stop - start)

    plot1 = Plotter(mo1)
    plot2 = Plotter(mo2)

    crepy_paris_dist = R_EARTH * ut.calc_circ_dist(np.radians(COORD_CREPY), np.radians(COORD_PARIS))
    mo1_mo2_dist = proj1.r * ut.calc_circ_dist(np.radians(ut.r_vec_to_coord(mo1.r_vec_arr[-1])),
                                          np.radians(ut.r_vec_to_coord(mo2.r_vec_arr[-1])))
    paris_mo1_dist = proj1.r * ut.calc_circ_dist(np.radians(COORD_PARIS),
                                          np.radians(ut.r_vec_to_coord(mo1.r_vec_arr[-1])))
    paris_mo2_dist = proj1.r * ut.calc_circ_dist(np.radians(COORD_PARIS),
                                                 np.radians(ut.r_vec_to_coord(mo2.r_vec_arr[-1])))

    print("Crepy coords:\t\t", COORD_CREPY, "deg")
    print("Paris coords:\t\t", COORD_PARIS, "deg")
    print("Crepy - Paris azimuth:\t", azimuth, "deg")
    print("Shooting direction:\t", th1, "deg")
    print("Crepy - Paris distance:\t", crepy_paris_dist, "meters")

    print("\t=== CORIOLIS OFF ===")
    print("Max height:\t", mo1.z_max, "m")
    print("Time:\t", mo1.tl, "s")
    print("Coords:\t", ut.r_vec_to_coord(mo1.r_vec_arr[-1]), "deg")
    print("Distance:\t", mo1.distance, "m")
    print("Angle:\t", 360 + np.rad2deg(mo1.th), "deg\tstart:\t", np.rad2deg(proj1.th0), "deg")
    print("Distance from Paris:\t", paris_mo1_dist, "m")

    print("\t=== CORIOLIS ON ===")
    print("Max height:\t", mo2.z_max, "m\tdelta =", mo2.z_max - mo1.z_max, "m")
    print("Time:\t", mo2.tl, "s")
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
    ax.scatter(crepy[0], crepy[1], crepy[2], c="r", marker="^", label=r"Crépy")
    ax.scatter(paris[0], paris[1], paris[2], c="g", marker="^", label=r"Paris")

    #   Marking of crash sites
    ax.scatter(plot1.crash[0], plot1.crash[1], plot1.crash[2], c="k", marker="x")
    ax.scatter(plot2.crash[0], plot2.crash[1], plot2.crash[2], c="m", marker="x")

    #   Plotting of chords, trajectory curves and surfaces
    u = np.radians(np.linspace(COORD_CREPY[1], COORD_PARIS[1], 100))
    v = np.radians(np.linspace(COORD_CREPY[0], COORD_PARIS[0], 100))
    crepy_paris_line = R_EARTH * np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])
    earth = R_EARTH * np.array([np.outer(np.cos(u), np.cos(v)),
                                np.outer(np.sin(u), np.cos(v)), np.outer(np.ones(np.size(u)),
                                                                         np.sin(v))])
    ax.plot(crepy_paris_line[0], crepy_paris_line[1], crepy_paris_line[2], label=r"Surface", alpha=0.5)
    ax.plot(plot1.line_3d[0], plot1.line_3d[1], plot1.line_3d[2], c="k", label=r"Without coriolis", linewidth=0.8)
    ax.plot(plot2.line_3d[0], plot2.line_3d[1], plot2.line_3d[2], c="m", label=r"With coriolis", linewidth=0.8)
    ax.plot_surface(earth[0], earth[1], earth[2], color='b', alpha=0.1)

    plt.legend()
    ax.view_init(azim=-45, elev=45)
    plt.savefig("3D_trajectory.pdf")
    plt.show()

    plt.figure()
    plt.plot(plot1.dist, plot1.height, label=r"Without coriolis", color="k")
    plt.plot(plot2.dist, plot2.height, label=r"With coriolis", color="m")
    plt.plot(0, 0, "r^", label=r"Crépy")
    plt.plot(crepy_paris_dist, 0, "g^",label=r"Paris")
    plt.xlabel(r"Distance", fontsize=16)
    plt.ylabel(r"Height, $z$", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("2D_trajectory.pdf")
    plt.show()

    plt.figure()
    plt.plot(plot1.long, plot1.lat, label=r"Without coriolis", color="k")
    plt.plot(plot2.long, plot2.lat, label=r"With coriolis", color="m")
    plt.plot(plot1.long[-1], plot1.lat[-1], "kx")
    plt.plot(plot2.long[-1], plot2.lat[-1], "mx")
    plt.plot(COORD_CREPY[1], COORD_CREPY[0], 'r^', label=r"Crépy")
    plt.plot(COORD_PARIS[1], COORD_PARIS[0], 'g^', label=r"Paris")
    #plt.xlim(left=2.3508, right=2.3509)#plt.xlim(left=2.350, right=2.351)    #   (left=2.3508, right=2.3509)
    #plt.ylim(bottom=48.8566, top=48.8567)#plt.ylim(bottom=48.856, top=48.857) #   (bottom=48.8566, top=48.8567)
    plt.xlabel(r"Longitude", fontsize=16)
    plt.ylabel(r"Latitude", fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig("lat_vs_long_trajectory.pdf")
    plt.show()