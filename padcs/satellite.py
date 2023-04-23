import numpy as np
from numpy import sin, cos, arcsin, arccos
import numpy.linalg as LA

import matplotlib.pyplot as plt

import padcs.utils as utils

class Satellite():

    def __init__(self, sim_time=0):
        self.sim_time = sim_time
        self.pos = np.empty([sim_time, 3])
        self.vel = np.empty([sim_time, 3])
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

    def to_attitude_wie(self, q, q_des, mu=1):
        self.q[0] = q
        self.q_des[0] = q_des
        sign = np.sign(q[3])

        for i in range(0, self.sim_time-1):
            self.q_e[i] = utils.calc_q_e(self.q[i], self.q_des[0])
            self.q_e[i] /= LA.norm(self.q_e[i])

            self.u[i] = np.matmul(np.matmul(utils.skew(-mu*self.w_sat[i]), self.J), self.w_sat[i]) \
                        - np.matmul(self.D, self.w_sat[i]) \
                        - sign * np.matmul(self.K, self.q_e[i, :3])
            self.dw_sat[i] = np.matmul(np.matmul(utils.skew(self.w_sat[i]), self.J), self.w_sat[i]) + self.u[i]
            self.dw_sat[i] = np.matmul(self.dw_sat[i], LA.inv(self.J))

            self.q_dot[i, :3] = 0.5 * np.matmul(utils.skew(self.w_sat[i]), self.q_e[i, :3]) \
                                + 0.5 * self.q_e[i, 3]*self.w_sat[i]
            self.q_dot[i, 3] = -0.5 * np.matmul(self.w_sat[i], self.q_e[i, :3])

            self.w_sat[i+1] = self.w_sat[i] + self.dw_sat[i]
            self.q[i+1] = self.q[i] + self.q_dot[i]
            self.q[i+1] /= LA.norm(self.q[i+1])

    def point_nadir(self, q, mu=1):
        self.q[0] = q
        sign = np.sign(q[3])    # update sign if q_des updates?

        for i in range(0, self.sim_time-1):
            self.q_des[i] = self._to_nadir(self.pos[i])
            self.q_e[i] = utils.calc_q_e(self.q[i], self.q_des[i])
            self.q_e[i] /= LA.norm(self.q_e[i])

            self.u[i] = np.matmul(np.matmul(utils.skew(-mu*self.w_sat[i]), self.J), self.w_sat[i]) \
                        - np.matmul(self.D, self.w_sat[i]) \
                        - sign * np.matmul(self.K, self.q_e[i, :3])
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

    def _to_nadir(self, pos):
        r = LA.norm(pos)

        q_x = -pos[0] / r
        q_y = -pos[1] / r
        q_z = -pos[2] / r
        q_w = 1

        q = np.array( [q_x, q_y, q_z, q_w] )
        q /= LA.norm(q)

        return q

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

    def set_inertia_matrix(self, J):
        self.J = np.array( [ [J[0][0], 0, 0],
                             [0, J[1][1], 0],
                             [0, 0, J[2][2]] ] )

    def set_orbit(self):

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

    def set_ang_vel(self, w):
        self.w_sat[0] = w

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

    def plot_orbit(self):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].set_title('x, y, z values over time')
        ax[0].plot(self.pos[:, 0], label='x')
        ax[0].plot(self.pos[:, 1], label='y')
        ax[0].plot(self.pos[:, 2], label='z')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[0].legend()

        ax[1].set_title('2d orbit planes')
        ax[1].plot(self.pos[:, 0], self.pos[:, 1], label='xy')
        ax[1].plot(self.pos[:, 0], self.pos[:, 2], label='xz')
        ax[1].plot(self.pos[:, 1], self.pos[:, 2], label='yz')
        ax[1].axis('equal')
        ax[1].set_xlabel('Position (km)')
        ax[1].set_ylabel('Position (km)')
        ax[1].legend()

        ax[2].set_title('velocity components')
        ax[2].plot(self.vel[:, 0], label='v_x')
        ax[2].plot(self.vel[:, 1], label='v_y')
        ax[2].plot(self.vel[:, 2], label='v_z')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Velocity (km/s)')
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_q(self, show_des=False, u_start=0):
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))

        ax[0, 0].set_title('Quaternion vs Time')
        ax[0, 0].plot(self.q[:, 0], label='q_x')
        ax[0, 0].plot(self.q[:, 1], label='q_y')
        ax[0, 0].plot(self.q[:, 2], label='q_z')
        ax[0, 0].plot(self.q[:, 3], label='q_w')
        if show_des:
            ax[0, 0].plot(self.q_des[:-1, 0], label='q_des_x', linestyle='dashed')
            ax[0, 0].plot(self.q_des[:-1, 1], label='q_des_y', linestyle='dashed')
            ax[0, 0].plot(self.q_des[:-1, 2], label='q_des_z', linestyle='dashed')
            ax[0, 0].plot(self.q_des[:-1, 3], label='q_des_w', linestyle='dashed')
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].set_ylabel('Quaternion Value')
        ax[0, 0].legend()

        ax[0, 1].set_title('Input Torque vs Time')
        ax[0, 1].plot(self.u[:-1, 0], label='u_x')
        ax[0, 1].plot(self.u[:-1, 1], label='u_y')
        ax[0, 1].plot(self.u[:-1, 2], label='u_z')
        ax[0, 1].set_xlabel('Time (s)')
        ax[0, 1].set_ylabel('Torque (Nm)')
        ax[0, 1].legend()

        ax[1, 0].set_title('Angular Velocity vs Time')
        ax[1, 0].plot(self.w_sat[:, 0], label='w_x')
        ax[1, 0].plot(self.w_sat[:, 1], label='w_y')
        ax[1, 0].plot(self.w_sat[:, 2], label='w_z')
        ax[1, 0].set_xlabel('Time (s)')
        ax[1, 0].set_ylabel('Angular Velocity (rad/s)')
        ax[1, 0].legend()

        ax[1, 1].set_title('Error Quaternion vs Time')
        ax[1, 1].plot(self.q_e[:-1, 0], label='q_e_x')
        ax[1, 1].plot(self.q_e[:-1, 1], label='q_e_y')
        ax[1, 1].plot(self.q_e[:-1, 2], label='q_e_z')
        ax[1, 1].plot(self.q_e[:-1, 3], label='q_e_w')
        ax[1, 1].set_xlabel('Time (s)')
        ax[1, 1].set_ylabel('Quaternion Value')
        ax[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def plot_u(self, start, end):
        x = range(start, end)
        plt.title('Input Torque vs Time')
        plt.plot(x, self.u[start:end, 0], label='u_x')
        plt.plot(x, self.u[start:end, 1], label='u_y')
        plt.plot(x, self.u[start:end, 2], label='u_z')
        plt.xlabel('Time (s)')
        plt.ylabel('Input Torque (Nm)')
        plt.legend()

        plt.tight_layout()
        plt.show()