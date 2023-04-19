import numpy as np

import matplotlib.pyplot as plt

from padcs.satellite import Satellite

# Example from Quaternion Feedback Regulator for 
# Spacecraft Eigenaxis Rotations (1988)

sat = Satellite(100)

J = [ [1200, 0, 0],
      [0, 2200, 0],
      [0, 0, 3100]]
J = np.array(J)

sat.set_inertia_matrix(J)

q_des = np.array( [0, 0, 0, 1] )
q_initial = np.array( [0.57, 0.57, 0.57, 0.159] )

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

sat.to_attitude(q_initial, q_des, mu=mu)

fig, ax = plt.subplots(2, 2, figsize=(16, 9))

print(sat.u[0])
print(sat.dw_sat[0])
print(sat.q_dot[0])

ax[0, 0].plot(sat.q[:, 0], label='q_x')
ax[0, 0].plot(sat.q[:, 1], label='q_y')
ax[0, 0].plot(sat.q[:, 2], label='q_z')
ax[0, 0].plot(sat.q[:, 3], label='q_w')
ax[0, 0].legend()

ax[0, 1].plot(sat.u[:-1, 0], label='u_x')
ax[0, 1].plot(sat.u[:-1, 1], label='u_y')
ax[0, 1].plot(sat.u[:-1, 2], label='u_z')
ax[0, 1].legend()

ax[1, 0].plot(sat.w_sat[:, 0], label='w_x')
ax[1, 0].plot(sat.w_sat[:, 1], label='w_y')
ax[1, 0].plot(sat.w_sat[:, 2], label='w_z')
ax[1, 0].legend()

plt.show()