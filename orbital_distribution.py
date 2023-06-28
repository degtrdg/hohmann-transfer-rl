import HohmannTransferEnv as hte
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from tqdm import tqdm

env = hte.HohmannTransferEnv()
env.reset()

earth = plt.Circle((0, 0), env.earth_radius, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)
minor_ticks = np.arange(-10*env.earth_radius, 10*env.earth_radius, env.earth_radius)
major_ticks = np.arange(-10*env.earth_radius, 10*env.earth_radius, 5*env.earth_radius)

for i in tqdm(range(1000)):
    a, e = om.random_orbit(env, max_a=4, max_c=3)
    trajectory = om.orbit_trajectory(a, e)
    plt.plot(trajectory[0,:], trajectory[1,:], 'darkorange', alpha=0.01)

ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(minor_ticks, minor=True)
plt.grid(color='lightgray',linestyle='--',which='minor',alpha=0.2)
plt.grid(color='lightgray',linestyle='--',which='major',alpha=0.5)
plt.show()
