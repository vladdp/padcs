import numpy as np
from numpy import sin, cos, arccos
import numpy.linalg as LA

import padcs.utils as utils

class Satellite():

    def __init__(self, sim_time=0):
        self.sim_time = sim_time
        self.pos = np.empty([sim_time, 3])
        self.vel = np.empty([sim_time, 3])
        self.ang = np.empty([sim_time, 3]) # (theta, phi, psi)
        self.w_sat = np.empty([sim_time, 3])
        self.w_dot = np.empty([sim_time, 3])
        self.q = np.empty([sim_time, 4])
        self.u = np.empty([sim_time, 3]) # input torque

    def change_to_attitude(self):

        self.u[0, 1] = 1

        # calculate ang acc from torque
        # self.w_dot[0] = np.matmul(self.u[0] - np.matmul(utils.skew(self.w_sat[0]), np.matmul(self.J, self.w_sat[0])), 
        #                   LA.inv(self.J))
        
        # calculate ang vel from ang acc
        # self.w_sat[1] += self.w_dot[0]

        for i in range(1, self.sim_time):
            self.w_dot[i] = np.matmul(self.u[i] - np.matmul(utils.skew(self.w_sat[i-1]), np.matmul(self.J, self.w_sat[i-1])), 
                            LA.inv(self.J))

            self.w_sat[i] += self.w_dot[i]
            
        
        print(self.w_dot[1])
        print(self.w_sat[0])
        print(self.w_sat[1])

    def set_desired_vector(self, desired):
        self.desired = desired

    def set_PID(self, p, i, d):
        self.pid = [p, i, d]

    def set_orbital_elements(self, a, e, i, raan, w, nu=0):
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.w = w
        self.nu = nu

        self.p = a * (1-e**2)
        self.t = 2 * np.pi * np.sqrt(self.a**3 / utils.MU)

    def set_quaternion(self):
        self.q[0, 0] = 1
        self.q[0, 1] = 0
        self.q[0, 2] = 0
        self.q[0, 3] = 0

        q_123 = [ [self.q[0, 0], -self.q[0, 3], self.q[0, 2]],
                  [self.q[0, 3], self.q[0, 0], -self.q[0, 1]],
                  [-self.q[0, 2], self.q[0, 1], self.q[0, 0]] ]
        
        q_dot = 0.5 * np.matmul(q_123, self.w_sat[0])

        print(q_dot)

    def set_inertia_matrix(self, J):
        self.J = J

    def set_pos(self):

        for i in range(self.sim_time):
            r = self.p / (1+self.e * cos(self.nu))
            self.pos[i, 0] = r * cos(self.nu)
            self.pos[i, 1] = r * sin(self.nu)
            self.pos[i, 2] = 0

            nup = np.sqrt(utils.MU / (self.p))
            self.vel[i][0] = nup * -sin(self.nu)
            self.vel[i][1] = nup * (self.e + cos(self.nu))
            self.vel[i][2] = 0

            nx = self.pos[i][0] + self.vel[i][0]
            ny = self.pos[i][1] + self.vel[i][1]
            nr = np.sqrt(nx**2 + ny**2)

            if ny > 0:
                self.nu = arccos(nx/nr)
            else:
                self.nu = 2 * np.pi - arccos(nx/nr)

            self.pos[i] = utils.rot_z(self.pos[i], self.w)
            self.pos[i] = utils.rot_x(self.pos[i], self.i)
            self.pos[i] = utils.rot_z(self.pos[i], self.raan)

            self.vel[i] = utils.rot_z(self.vel[i], self.w)
            self.vel[i] = utils.rot_x(self.vel[i], self.i)
            self.vel[i] = utils.rot_z(self.vel[i], self.raan)
        
    def set_angle(self):

        self.ang[0, 0] = 0
        self.ang[0, 1] = self.i
        self.ang[0, 2] = np.pi

        for i in range(1, self.sim_time):
            self.ang[i, 0] = self.ang[i-1, 0] + self.w_sat[i, 0]
            self.ang[i, 1] = self.ang[i-1, 1] + self.w_sat[i, 1]
            self.ang[i, 2] = self.ang[i-1, 2] + self.w_sat[i, 2]

    def set_ang_vel(self):
        # For Nadir pointing assuming s/c starts pointing at Earth.
        self.w_0 = (2 * np.pi) / self.t

        self.w_sat[:, 0] = 0
        self.w_sat[:, 1] = 0
        self.w_sat[:, 2] = self.w_0

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
        if not isinstance(r, np.ndarray):
            r = np.asarray(r, dtype=np.float64)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float64)

        self.h = np.cross(r, v)             
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
        print(np.degrees(self.nu))