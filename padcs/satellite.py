import numpy as np
from numpy import sin, cos, arcsin, arccos
import numpy.linalg as LA

import padcs.utils as utils

class Satellite():

    def __init__(self, sim_time=0):
        self.sim_time = sim_time
        self.pos = np.empty([sim_time, 3])
        self.vel = np.empty([sim_time, 3])
        self.ang = np.empty([sim_time, 3])      # (theta, phi, psi)
        self.w_sat = np.empty([sim_time, 3])
        self.dw_sat = np.empty([sim_time, 3])
        self.q = np.empty([sim_time, 4])
        self.q_dot = np.empty([sim_time, 4])
        self.q_des = np.empty([sim_time, 4])
        self.q_e = np.empty([sim_time, 4])
        self.u = np.empty([sim_time, 3])        # input torque

        self.c_q = np.empty([sim_time])
        self.c_q[0] = 1

        self.e_sum = 0

        self.J_theta_dot = np.empty([sim_time, 3])

    def to_attitude(self, q, q_des, mu=1):
        self.q[0] = q
        self.q_des[0] = q_des
        self.q_e[0] = utils.calc_q_e(self.q[0], self.q_des[0])
        self.q_e[0] /= LA.norm(self.q_e[0])

        # u = -mu*Omega*J*w - D*w - sign(q4(0))*K*q
        self.u[0] = np.matmul(np.matmul(utils.skew(-mu*self.w_sat[0]), self.J), self.w_sat[0]) \
                    - np.matmul(self.D, self.w_sat[0]) \
                    - np.sign(q[3]) * np.matmul(self.K, self.q_e[0, :3])

        # J_dw = Omega*J*w + u
        self.dw_sat[0] = np.matmul(np.matmul(utils.skew(self.w_sat[0]), self.J), self.w_sat[0]) + self.u[0]
        # dw = ^ / J
        self.dw_sat[0] = np.matmul(self.dw_sat[0], LA.inv(self.J))

        # dq = 0.5*Omega*q + 0.5*q_4*w
        self.q_dot[0, :3] = 0.5 * np.matmul(utils.skew(self.w_sat[0]), self.q_e[0, :3]) \
                            + 0.5 * self.q_e[0, 3]*self.w_sat[0]
        # dw_4 = -0.5*w*q
        self.q_dot[0, 3] = -0.5 * np.matmul(self.w_sat[0], self.q_e[0, :3])

        self.w_sat[1] = self.w_sat[0] + self.dw_sat[0]
        self.q[1] = self.q[0] + self.q_dot[0]
        self.q[1] /= LA.norm(self.q[1])

        for i in range(1, self.sim_time-1):
            self.q_e[i] = utils.calc_q_e(self.q[i], self.q_des[0])
            self.q_e[i] /= LA.norm(self.q_e[i])
            self.u[i] = np.matmul(np.matmul(utils.skew(-mu*self.w_sat[i]), self.J), self.w_sat[i]) \
                    - np.matmul(self.D, self.w_sat[i]) \
                    - np.sign(q[3]) * np.matmul(self.K, self.q_e[i, :3])
            self.dw_sat[i] = np.matmul(np.matmul(utils.skew(self.w_sat[i]), self.J), self.w_sat[i]) + self.u[i]
            self.dw_sat[i] = np.matmul(self.dw_sat[i], LA.inv(self.J))

            self.q_dot[i, :3] = 0.5 * np.matmul(utils.skew(self.w_sat[i]), self.q_e[i, :3]) \
                            + 0.5 * self.q_e[i, 3]*self.w_sat[i]
            self.q_dot[i, 3] = -0.5 * np.matmul(self.w_sat[i], self.q_e[i, :3])

            self.w_sat[i+1] = self.w_sat[i] + self.dw_sat[i]
            self.q[i+1] = self.q[i] + self.q_dot[i]
            self.q[i+1] /= LA.norm(self.q[i+1])


    def set_gains(self, k, d):
        self.k = k
        self.d = d

        self.K = np.array( k*self.J )
        self.D = np.array( d*self.J )

    def nadir(self):

        # self.q_des[0] = self._to_nadir(self.pos[0])
        self.q_des[0] = [0, -1, 0, 0]

        # self.q_e[0] = self.q_des[0] - self.q[0]
        self.q_e[0] = utils.calc_q_e(self.q_des[0], self.q[0])
        self.e_sum += self.q_e[0]

        self.J_theta_dot[0] = self.pid[0]*self.q_e[0, 1:] + self.pid[2]*(self.q_e[0, 1:]) \
                    + self.pid[1]*self.e_sum[1:]

        # self.dw_sat[0] = np.matmul( self.J_theta_dot[0], LA.inv(self.J))
        self.dw_sat[0] = self.J_theta_dot[0] / (LA.norm(self.J))
        self.w_sat[0] = self.dw_sat[0]

        self.q_dot[0] = utils.calc_q_dot(self.q[0], self.w_sat[0])

        self.q[1] = self.q[0] + self.q_dot[0]
        self.q[1] /= LA.norm(self.q[1])

        for i in range(1, self.sim_time-1):
            # self.q_des[i] = self._to_nadir(self.pos[i])
            self.q_des[i] = [0, -1, 0, 0]
            # self.q_e[i] = self.q_des[i] - self.q[i]
            self.q_e[i] = utils.calc_q_e(self.q_des[i], self.q[i])
            self.e_sum += self.q_e[i]

            self.J_theta_dot[i] = self.pid[0]*self.q_e[i, 1:] + self.pid[2]*(self.q_e[i, 1:]-self.q_e[i-1, 1:]) \
                    + self.pid[1]*self.e_sum[1:]
            # self.dw_sat[i] = np.matmul( control, LA.inv(self.J) )
            self.dw_sat[i] = self.J_theta_dot[i] / LA.norm(self.J)
            self.w_sat[i] = self.w_sat[i-1] + self.dw_sat[i]

            self.q_dot[i] = utils.calc_q_dot(self.q[i], self.w_sat[i])
            self.q[i+1] = self.q[i] + self.q_dot[i]
            self.q[i+1] /= LA.norm(self.q[i+1])

    def _to_nadir(self, pos):
        r = LA.norm(pos)

        q_w = 0
        q_x = -pos[0] / r
        q_y = -pos[1] / r
        q_z = -pos[2] / r

        return [q_w, q_x, q_y, q_z]

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

    def set_ang_vel(self, w):
        self.w_sat[0] = w

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