import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr
import numpy as np
from SimpleBurnEnv import SimpleBurnEnv as sbe
import matplotlib.pyplot as plt
from tqdm import tqdm

env = sbe()

earth = plt.Circle((0, 0), env.tbr.r1, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)
minor_ticks = np.arange(-5*env.tbr.r1, 5*env.tbr.r1, env.tbr.r1)
major_ticks = np.arange(-5*env.tbr.r1, 5*env.tbr.r1, 5*env.tbr.r1)

for i in tqdm(range(1000)):
    env.reset()
    e, a = env.target
    trajectory = om.orbit_trajectory(e, a)
    plt.plot(trajectory[0,:], trajectory[1,:], 'darkorange', alpha=0.01)

trajectory = om.orbit_trajectory(env.state[1:3], env.state[3])
plt.plot(trajectory[0,:], trajectory[1,:], 'cyan', alpha=1)

ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(minor_ticks, minor=True)
plt.grid(color='lightgray',linestyle='--',which='minor',alpha=0.2)
plt.grid(color='lightgray',linestyle='--',which='major',alpha=0.5)
plt.xlim(-4*env.tbr.r1, 4*env.tbr.r1)
plt.ylim(-4*env.tbr.r1, 4*env.tbr.r1)
plt.title("Possible Target Orbits")
plt.show()
