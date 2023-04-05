import numpy as np
from numpy import sin, cos, arccos
import numpy.linalg as LA

import padcs.utils as utils

class Satellite():

    def __init__(self):
        self.F_b = []       # Angle of the satellite relative to its local frame.
        pass

    def set_r_and_v_from_elements(self, p, e, i, G, w, nu):
        # self.p = a * (1-e**2)
        self.p = p

        self.radius = self.p / (1+e*cos(nu))
        self.r = np.empty(3)
        self.v = np.empty(3)
        self.r[0] = self.radius*cos(nu)
        self.r[1] = self.radius*sin(nu)
        self.r[2] = 0
        self.v[0] = np.sqrt(utils.MU/self.p) * -sin(nu)
        self.v[1] = np.sqrt(utils.MU/self.p) * (e+cos(nu))
        self.v[2] = 0


    def set_orbit_from_r_and_v(self, r, v):
        """ r and v must be numpy arrays. """
        # TODO convert r and v into numpy arrays if they are not.

        if not isinstance(r, np.ndarray):
            r = np.asarray(r, dtype=np.float64)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float64)

        self.h = np.cross(r, v)             # angular momentum vector
        self.p = LA.norm(self.h)**2/utils.MU
        self.n = np.cross(utils.K, self.h)  # node vector (line of nodes)
        self.e = (1/utils.MU) * ( (LA.norm(v)**2 - utils.MU/LA.norm(r))*r - np.dot(r, v)*v )
        self.i = arccos(self.h[2]/LA.norm(self.h))
        self.G = arccos(self.n[0]/LA.norm(self.n))
        self.w = arccos(np.dot(self.n, self.e) / (LA.norm(self.n)*LA.norm(self.e)))
        self.v_0 = arccos(np.dot(self.e, r) / (LA.norm(self.e)*LA.norm(r)))
        self.u_0 = arccos(np.dot(self.n, r) / (LA.norm(self.n)*LA.norm(r)))
        self.l_0 = self.G + self.u_0

    def print_elements(self):
        print(self.h)
        print(self.p)
        print(self.n)
        print(self.e)
        print(np.degrees(self.i))
        print(np.degrees(self.G))
        print(np.degrees(self.w))
        print(np.degrees(self.v_0))
        print(np.degrees(self.u_0))
        print(np.degrees(self.l_0))
