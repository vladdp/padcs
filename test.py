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
raan = np.radians(40)
w = np.radians(10)
nu = np.radians(0)

sat.set_orbital_elements(a, e, i, raan, w, nu)

# print(sat.t)

sat.set_pos()

plt.plot(sat.pos[:, 0])
plt.plot(sat.pos[:, 1])
plt.plot(sat.pos[:, 2])

plt.show()