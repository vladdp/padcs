import numpy as np
from numpy import sin, cos


I = [1, 0, 0]
J = [0, 1, 0]
K = [0, 0, 1]

MU = 3.986 * (10 ** 5)          # Gravitational Parameter of Earth
RHO = 4.5 * (10 ** (-6))        # Solar Radiation Pressure Torque magnitude near Earth


def rot_x(xyz, theta):
    R_x = [ [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)] ]
    return np.matmul(R_x, xyz)

def rot_y(xyz, theta):
    R_y = [ [cos(theta), 0, sin(theta)],
          [0, 1, 0],
          [-sin(theta), 0, cos(theta)] ]
    return np.matmul(R_y, xyz)

def rot_z(xyz, theta):
    R_z = [ [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0], 
            [0, 0, 1] ]
    return np.matmul(R_z, xyz)

def rot_CbG(xyz, phi, theta, psi):
    CbG = [ [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
            [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta)],
            [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)] ]

    return np.matmul(CbG, xyz)

def get_dist(a, b):
    return np.linalg.norm(a-b)