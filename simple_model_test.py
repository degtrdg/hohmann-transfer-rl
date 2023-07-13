from stable_baselines3 import DQN
from stable_baselines3 import PPO
import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from SimpleBurnEnv import SimpleBurnEnv as sbe


env = sbe()
env.reset()

model_dir = "models/simple/DQN"
model = PPO.load(f'{model_dir}/75000')

# Initial orbit
trajectory = om.orbit_trajectory(env.orbit_state[1:3], env.orbit_state[3])
plt.plot(trajectory[0,:], trajectory[1,:], 'green', alpha=1)
# Target orbit
trajectory = om.orbit_trajectory(env.target[0], env.target[1])
plt.plot(trajectory[0,:], trajectory[1,:], 'blue', alpha=1)

positions = np.empty((1000,2))
burns = []
for i in range(1000):
    action, _states = model.predict(env.state)
    if action == 1:
        burns.append(env.ivp_state[0:2])
    step = env.step(action)
    if step[2] or step[3]:
        positions = positions[0:i,:]
        plt.scatter(env.ivp_state[0], env.ivp_state[1], color='red', marker='x', s=40)
        break
    positions[i,:] = env.ivp_state[0:2]
burns = np.asarray(burns)
plt.plot(positions[:,0], positions[:,1], 'darkorange', alpha=1)
plt.scatter(burns[:,0], burns[:,1], color='k', marker='x', linewidth=.5, s=10)

# Final orbit
trajectory = om.orbit_trajectory(env.orbit_state[1:3], env.orbit_state[3])
print(env.target_a, env.state[3]/env.tbr.r1)
print(np.linalg.norm(env.target[0]), np.sqrt(env.state[1]**2 + env.state[2]**2))
plt.plot(trajectory[0,:], trajectory[1,:], 'cyan', alpha=1)
plt.show()
