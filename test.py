import numpy as np

from padcs.satellite import Satellite

sat = Satellite()

# r = [12756.2, 0, 0]
# v = [0, 7.90537, 0]
r = [8750, 5100, 0]
v = [-3, 5.2, 5.9]

sat.set_orbit_from_r_and_v(r, v)
# sat.print_elements()

p = 14351
e = 0.5
i = np.radians(45)
G = np.radians(30)
w = np.radians(0)
v_0 = np.radians(0)

sat.set_r_and_v_from_elements(p, e, i, G, w, v_0)
print(sat.r)
print(sat.v)
