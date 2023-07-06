from SimpleBurnEnv import SimpleBurnEnv as sbe
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from tqdm import tqdm

env = sbe()
env.reset()

thetas = [i*2*np.pi for i in [0, 1/4, 1/2, 3/4]]

# product of 10,000 is a good time to run for
# decrease t to increase computation speed
t = 2000
dt = 5

earth = plt.Circle((0, 0), env.tbr.r1, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)

positions = np.empty((t,2))
env.reset()
for i in tqdm(range(t)):
    positions[i] = env.ivp_state[:2]
    env.step(0, dt=dt)

s = env.state
cm = plt.get_cmap('gist_rainbow')
for (i,v) in enumerate(thetas):
    env.reset(theta=v)
    e, a = env.target
    trajectory = om.orbit_trajectory(e, a)
    plt.plot(trajectory[0,:], trajectory[1,:], color=cm(1.*i/len(thetas)))
plt.plot(positions[:, 0], positions[:, 1], color='black')
plt.grid(color='lightgray',linestyle='--')
plt.title("Example Targets")
plt.show()
