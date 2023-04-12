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
i = np.radians(30)
raan = np.radians(0)
w = np.radians(0)
nu = np.radians(0)

sat.set_orbital_elements(a, e, i, raan, w, nu)

# print(sat.t)

sat.set_pos()


# plt.plot(sat.pos[:, 0], label='x')
# plt.plot(sat.pos[:, 1], label='y')
# plt.plot(sat.pos[:, 2], label='z')

sat.set_ang_vel()
sat.set_angle()

# ang velocity about z axis to maintain nadir
print(sat.w_0)

# print(sat.ang)
# print(sat.w_sat)

# plt.plot(sat.ang[:, 0], label='theta')
# plt.plot(sat.ang[:, 1], label='phi')
# plt.plot(sat.ang[:, 2], label='psi')

plt.plot(sat.w_sat[:, 0], label='w_x')
plt.plot(sat.w_sat[:, 1], label='w_y')
plt.plot(sat.w_sat[:, 2], label='w_z')

plt.legend()

plt.show()