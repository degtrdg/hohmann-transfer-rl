from SimpleBurnEnv import SimpleBurnEnv as sbe
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om

env = sbe()
env.reset()
print(np.linalg.norm(env.target[0]))

# trajectory = om.orbit_trajectory(env.state[1:3], env.state[3])
# plt.plot(trajectory[0,:], trajectory[1,:], 'darkorange', alpha=1)

# plt.scatter(env.ivp_state[0], env.ivp_state[1], color='k', marker='x')
steps = 750
positions = np.empty((steps,2))
for i in range(steps):
    env.step(0)
    positions[i,:] = env.ivp_state[0:2]
plt.plot(positions[:,0], positions[:,1], 'darkorange', alpha=1)
# plt.scatter(env.ivp_state[0], env.ivp_state[1], color='k', marker='x')

plt.show()

