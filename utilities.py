__author__ = 'Jullan'
# -*- coding: utf-8 -*-
#Made by Jullan
import numpy as np

def calc_deg(deg, mins, secs):
    return deg + mins/60 + secs/3600

def r_vec_to_coord(r_vec):
    lambd = np.pi / 2 - r_vec[0] / r_vec[2]
    phi = r_vec[1]/(np.cos(lambd)*r_vec[2])
    return np.rad2deg(np.array([lambd, phi]))

def r_vec_to_cartesian(r_vec):
    lambd = np.pi/2-r_vec[0]/r_vec[2]
    phi = r_vec[1]/(np.cos(lambd)*r_vec[2])
    return r_vec[2] * np.array([np.cos(lambd) * np.cos(phi),
                                np.cos(lambd) * np.sin(phi),
                                np.sin(lambd)])

def coord_to_cartesian(coord): #    Calculates cartesian coordinates in a unit sphere.
    coord_rad = np.radians(coord)
    return np.array([np.cos(coord_rad[0]) * np.cos(coord_rad[1]),
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

#       Calculation of the great circle distance.
def calc_circ_dist(coord1, coord2):
    return np.arccos(np.sin(coord1[0]) * np.sin(coord2[0]) + np.cos(coord1[0]) * np.cos(coord2[0]) * np.cos(coord2[1]-coord1[1]))

def interpolate_xl( x0, x1, y0, y1):
    return (x0 - y0 * x1 / y1) / (- y0 / y1 + 1)