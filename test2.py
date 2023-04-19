import numpy as np

import matplotlib.pyplot as plt

from padcs.satellite import Satellite
from padcs.disturbances import Disturbances


sim_time = 10000

sat = Satellite(sim_time)

# J = [ [1200, 100, -200],
#       [100, 2200, 300],
#       [-200, 300, 3100]]

J = [ [1200, 0, 0],
      [0, 2200, 0],
      [0, 0, 3100]]

a = 7000
e = 0
i = 0
raan = 0
w = 0
nu = 0

sat.set_orbital_elements(a, e, np.radians(i), 
    np.radians(raan), np.radians(w), np.radians(nu))

# print(sat.t)
sat.set_inertia_matrix(J)
sat.set_pos()

sat.set_PID(0.0005, 0, 1)

sat.nadir()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

ax[0, 0].plot(sat.q_des[:-1, 1], '--', label='q_des_x')
ax[0, 0].plot(sat.q_des[:-1, 2], '--', label='q_des_y')
ax[0, 0].plot(sat.q_des[:-1, 3], '--', label='q_des_z')
ax[0, 0].plot(sat.q[:, 1], ':', label='q_x')
ax[0, 0].plot(sat.q[:, 2], ':', label='q_y')
ax[0, 0].plot(sat.q[:, 3], ':', label='q_z')
ax[0, 0].plot(sat.q[:, 0], ':', label='q_w')
ax[0, 0].legend()

ax[0, 1].plot(sat.dw_sat[:-1, 0], label='dw_x')
ax[0, 1].plot(sat.dw_sat[:-1, 1], label='dw_y')
ax[0, 1].plot(sat.dw_sat[:-1, 2], label='dw_z')
ax[0, 1].legend()

ax[0, 2].plot(sat.w_sat[:-1, 0], label='w_x')
ax[0, 2].plot(sat.w_sat[:-1, 1], label='w_y')
ax[0, 2].plot(sat.w_sat[:-1, 2], label='w_z')
ax[0, 2].legend()

ax[1, 0].plot(sat.q_dot[:-1, 0], label='q_dot_w')
ax[1, 0].plot(sat.q_dot[:-1, 1], label='q_dot_x')
ax[1, 0].plot(sat.q_dot[:-1, 2], label='q_dot_y')
ax[1, 0].plot(sat.q_dot[:-1, 3], label='q_dot_z')
ax[1, 0].legend()

ax[1, 1].plot(sat.q_e[:-1, 0], label='q_e_w')
ax[1, 1].plot(sat.q_e[:-1, 1], label='q_e_x')
ax[1, 1].plot(sat.q_e[:-1, 2], label='q_e_y')
ax[1, 1].plot(sat.q_e[:-1, 3], label='q_e_z')
ax[1, 1].legend()

q_abs = np.empty([sim_time])

for i in range(len(sat.q)):
    q_abs[i] = np.linalg.norm(sat.q[i])

# ax[1, 2].plot(q_abs, label='q_abs')
ax[1, 2].plot(sat.J_theta_dot[:-1, 0], label='J_0_d2_x')
ax[1, 2].plot(sat.J_theta_dot[:-1, 1], label='J_0_d2_y')
ax[1, 2].plot(sat.J_theta_dot[:-1, 2], label='J_0_d2_z')
# ax[1, 2].plot(sat.J_theta_dot[:-1, 3], label='J_0_d2_z')
ax[1, 2].legend()

plt.legend()
plt.show()