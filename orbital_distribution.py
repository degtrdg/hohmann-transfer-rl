import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr
from tqdm import tqdm

tbr = tbr()

earth = plt.Circle((0, 0), tbr.r1, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)
minor_ticks = np.arange(-10*tbr.r1, 10*tbr.r1, tbr.r1)
major_ticks = np.arange(-10*tbr.r1, 10*tbr.r1, 5*tbr.r1)

for i in tqdm(range(1000)):
    e, a = om.random_orbit(tbr, max_a=4, min_a=2, max_c=2)
    trajectory = om.orbit_trajectory(a, e)
    plt.plot(trajectory[0,:], trajectory[1,:], 'darkorange', alpha=0.01)

ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(minor_ticks, minor=True)
plt.grid(color='lightgray',linestyle='--',which='minor',alpha=0.2)
plt.grid(color='lightgray',linestyle='--',which='major',alpha=0.5)
plt.xlim(-6*tbr.r1, 6*tbr.r1)
plt.ylim(-6*tbr.r1, 6*tbr.r1)
plt.show()
