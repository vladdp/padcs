import numpy as np

import matplotlib.pyplot as plt

from padcs.satellite import Satellite
from padcs.disturbances import Disturbances


sim_time = 10000

sat = Satellite(sim_time)

J = [ [1200, 100, -200],
      [100, 2200, 300],
      [-200, 300, 3100]]

a = 10000
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

sat.set_PID(0.0005, 1, 1)

sat.nadir()


fig, ax = plt.subplots(2, 3, figsize=(16, 9))

ax[0, 0].plot(sat.q_des[:, 1], label='q_des_x')
ax[0, 0].plot(sat.q_des[:, 2], label='q_des_y')
ax[0, 0].plot(sat.q_des[:, 3], label='q_des_z')
ax[0, 0].plot(sat.q[:, 1], label='q_x')
ax[0, 0].plot(sat.q[:, 2], label='q_y')
ax[0, 0].plot(sat.q[:, 3], label='q_z')
ax[0, 0].legend()

ax[0, 1].plot(sat.dw_sat[:, 0], label='dw_x')
ax[0, 1].plot(sat.dw_sat[:, 1], label='dw_y')
ax[0, 1].plot(sat.dw_sat[:, 2], label='dw_z')
ax[0, 1].legend()

ax[0, 2].plot(sat.w_sat[:, 0], label='w_x')
ax[0, 2].plot(sat.w_sat[:, 1], label='w_y')
ax[0, 2].plot(sat.w_sat[:, 2], label='w_z')
ax[0, 2].legend()

ax[1, 0].plot(sat.q_dot[:, 0], label='q_dot_w')
ax[1, 0].plot(sat.q_dot[:, 1], label='q_dot_x')
ax[1, 0].plot(sat.q_dot[:, 2], label='q_dot_y')
ax[1, 0].plot(sat.q_dot[:, 3], label='q_dot_z')
ax[1, 0].legend()

ax[1, 1].plot(sat.error[:, 0], label='error_w')
ax[1, 1].plot(sat.error[:, 1], label='error_x')
ax[1, 1].plot(sat.error[:, 2], label='error_y')
ax[1, 1].plot(sat.error[:, 3], label='error_z')
ax[1, 1].legend()

plt.legend()
plt.show()