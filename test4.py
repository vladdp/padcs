import numpy as np

from padcs.satellite import Satellite

# Testing nadir pointing given an orbit

sim_time = 100

sat = Satellite(sim_time)

J = [ [1200, 100, -200],
      [100, 2200, 300],
      [-200, 300, 3100]]

q_initial = [1, 0, 0, 1]

k = 0.05
d = 0.316
mu = 0.9
w_0 = [0.01, 0.01, 0.01]

sat.set_inertia_matrix(J)
sat.set_gains(k, d)
sat.set_ang_vel(w_0)

a = 7000
e = 0.5
i = 0
raan = 0
w = 0
nu = 0

sat.set_orbital_elements(a, e, i, raan, w, nu)
sat.set_orbit()
# sat.plot_orbit()

sat.point_nadir(q_initial, mu=mu)
sat.plot_q(show_des=True, norm_u=False)
# sat.plot_u(1000, sim_time)
