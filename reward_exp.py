import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr

tbr = tbr()


def reward_exp(delta_e, delta_a):
    return np.exp(-(delta_a/((4-1.3)*tbr.r1))**2) * np.exp(-(delta_e/.5)**2)

def reward_space(reward_func, tbr):
    e_difference = np.linspace(0, 0.5, 25)
    a_difference = np.linspace(0, (4-1.3)*tbr.r1, 25)

    e, a = np.meshgrid(e_difference, a_difference)
    r = reward_func(e, a)
    r = r[:-1, :-1]
    return e, a, r

e, a, r = reward_space(reward_exp, tbr)
r_min, r_max = -np.abs(r).max(), np.abs(r).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(e, a, r, cmap='RdBu', vmin=r_min, vmax=r_max)
ax.set_title('reward over difference in eccentricity and semi-major axis')
ax.set_xlabel('delta eccentricity')
ax.set_ylabel('delta semi-major axis')
# set the limits of the plot to the limits of the data
ax.axis([e.min(), e.max(), a.min(), a.max()])
fig.colorbar(c, ax=ax)

e0 = np.array([0, 0])
a0 = tbr.r1*1.3
ediff = []
adiff = []
for i in range(500):
    orbit = om.random_orbit(tbr, max_a=4, min_a=2, max_c=2)
    ediff.append(np.linalg.norm(e0 - orbit[0]))
    adiff.append(orbit[1]-a0)

ax.plot(ediff, adiff, 'x', color='black', markersize=2)
plt.show()





