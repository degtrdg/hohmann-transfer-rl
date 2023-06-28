import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr

tbr = tbr()

mags = []
sma = []
for i in range(10000):
    e, a = om.random_orbit(tbr, max_a=4, min_a=2, max_c=2)
    mags.append(np.linalg.norm(e))
    sma.append(a)

plt.subplot(2,1,1)
plt.hist(mags, bins=100)
plt.title("Eccentricity")
plt.subplot(2,1,2)
plt.hist(sma, bins=100)
plt.xlabel("Semi-major axis")
plt.xticks(np.arange(tbr.r1*2, tbr.r1*4, tbr.r1))
plt.show()