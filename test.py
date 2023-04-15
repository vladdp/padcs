import numpy as np

import matplotlib.pyplot as plt

from padcs.satellite import Satellite
from padcs.disturbances import Disturbances


sim_time = 10000

sat = Satellite(sim_time)
dist = Disturbances()


r = [8750, 5100, 0]
v = [-3, 5.2, 5.9]

J = [ [1200, 100, -200],
      [100, 2200, 300],
      [-200, 300, 3100]]

a = 10000
e = 0.8
i = 30
raan = 100
w = 0
nu = 0

sat.set_orbital_elements(a, e, np.radians(i), 
    np.radians(raan), np.radians(w), np.radians(nu))

# print(sat.t)
sat.set_inertia_matrix(J)
sat.set_pos()

sat.set_ang_vel()
sat.nadir()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
title = ( f"a=%s, e=%s, i=%s, raan=%s, w=%s, nu=%s" % 
         (a, e, i, raan, w, nu) )
fig.suptitle(title)

ax[0, 0].set_title('x, y, z values over time')
ax[0, 0].plot(sat.pos[:, 0], label='x')
ax[0, 0].plot(sat.pos[:, 1], label='y')
ax[0, 0].plot(sat.pos[:, 2], label='z')
ax[0, 0].legend()

ax[0, 1].set_title('2d orbit planes')
ax[0, 1].plot(sat.pos[:, 0], sat.pos[:, 1], label='xy')
ax[0, 1].plot(sat.pos[:, 0], sat.pos[:, 2], label='xz')
ax[0, 1].plot(sat.pos[:, 1], sat.pos[:, 2], label='yz')
ax[0, 1].axis('equal')
ax[0, 1].legend()

ax[0, 2].set_title('velocity components')
ax[0, 2].plot(sat.vel[:, 0], label='v_x')
ax[0, 2].plot(sat.vel[:, 1], label='v_y')
ax[0, 2].plot(sat.vel[:, 2], label='v_z')
ax[0, 2].legend()


# print(sat.vel[:, 2])

sat.set_ang_vel()
sat.set_angle()

# ang velocity about z axis to maintain nadir
# print("w_0:\t", sat.w_0)
# print("w_sat:\t", sat.w_sat[0])

# print(sat.ang)
# print(sat.w_sat)

ax[1, 0].set_title('s/c angle relative to inertial')
ax[1, 0].plot(sat.ang[:, 0], label='theta')
ax[1, 0].plot(sat.ang[:, 1], label='phi')
ax[1, 0].plot(sat.ang[:, 2], label='psi')
ax[1, 0].set_xlabel('time (s)')
ax[1, 0].set_ylabel('angle (rad)')
ax[1, 0].legend()

# sat.set_quaternion(0, 0, 0, 1)
# print("q:\t", sat.q[0])
# print("|q|:\t", np.linalg.norm(sat.q[0]))

# sat.set_quaternion()

ax[1, 1].set_title("angular velocity components")
ax[1, 1].plot(sat.w_sat[:, 0], label='w_x')
ax[1, 1].plot(sat.w_sat[:, 1], label='w_y')
ax[1, 1].plot(sat.w_sat[:, 2], label='w_z')
ax[1, 1].set_xlabel('time (s)')
ax[1, 1].legend()

ax[1, 2].set_title("angular acceleraction components")
ax[1, 2].plot(sat.dw_sat[:, 0], label='w_dot_x')
ax[1, 2].plot(sat.dw_sat[:, 1], label='w_dot_y')
ax[1, 2].plot(sat.dw_sat[:, 2], label='w_dot_z')
ax[1, 2].set_xlabel('time (s)')
ax[1, 2].legend()

# plt.show()