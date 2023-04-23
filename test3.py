import numpy as np

from padcs.satellite import Satellite

# Example from Quaternion Feedback Regulator for 
# Spacecraft Eigenaxis Rotations (1988)

sat = Satellite(200)

J = [ [1200, 0, 0],
      [0, 2200, 0],
      [0, 0, 3100]]

sat.set_inertia_matrix(J)

q_des = np.array( [1, 0, 0, 0.0] )
q_des /= np.linalg.norm(q_des)

q_initial = np.array( [0.57, 0.57, 0.57, 0.159] )
q_initial /= np.linalg.norm(q_initial)

# Settling time is assumed as 50 s.
t_s = 50

# Critically damped response (zeta = 1)
w_n = 0.158
k = 0.05
d = 0.316

sat.set_gains(k, d)

mu = 0.9
w_0 = [0.01, 0.01, 0.01]

sat.set_ang_vel(w_0)

sat.to_attitude_wie(q_initial, q_des, mu=mu)

# print(sat.u[0])
# print(sat.dw_sat[0])
# print(sat.q_dot[0])

sat.plot_q()
