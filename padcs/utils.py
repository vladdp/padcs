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

def skew(a):

    a_skew = [ [0, -a[2], a[1]],
               [a[2], 0, -a[0]],
               [-a[1], a[0], 0] ]

    return a_skew

def mult_q(p, q):
    # Can replace by Eq. 3.63 from Yang

    pv = np.array( (p[1], p[2], p[3]) )
    qv = np.array( (q[1], q[2], q[3]) )
    scalar = p[0] + q[0] - np.dot(pv, qv)
    vector = p[0]*qv + q[0]*pv + np.matmul(skew(pv), qv)

    return np.array( (scalar, vector[0], vector[1], vector[2]) )

def inv_q(q):
    return np.array( (q[0], -q[1], -q[2], -q[3]) )

def calc_q_dot(q, w):

    q = [ [q[0], -q[1], -q[2], -q[3]],
          [q[1], q[0], -q[3], q[2]],
          [q[2], q[3], q[0], -q[1]],
          [q[3], -q[2], q[1], q[0]] ]
    
    w = [0, w[0], w[1], w[2]]

    return 0.5 * np.matmul(q, w)

def calc_q_dot_3(q, w):

    q = [ [0, -q[3], q[2]],
          [q[3], 0, -q[1]],
          [-q[2], q[1], 0]]

    return 0.5 * np.matmul(q, w)

def calc_q_e(q, des):

    des = [ [des[3], des[2], -des[1], -des[0]],
            [-des[2], des[3], des[0], -des[1]],
            [des[1], -des[0], des[3], -des[2]],
            [des[0], des[1], des[2], des[3]] ]

    return np.matmul(des, q)

def get_dist(a, b):
    return np.linalg.norm(a-b)