import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
import HohmannTransferEnv as hte

env = hte.HohmannTransferEnv()
env.reset()

mags = []
sma = []
for i in range(10000):
    a, e = om.random_orbit(env, max_a=4, max_c=3)
    mags.append(np.linalg.norm(e))
    sma.append(a)

plt.subplot(2,1,1)
plt.hist(mags, bins=100)
plt.title("Eccentricity")
plt.subplot(2,1,2)
plt.hist(sma, bins=100)
plt.xlabel("Semi-major axis")
plt.xticks(np.arange(env.earth_radius*2, env.earth_radius*4, env.earth_radius))
plt.show()