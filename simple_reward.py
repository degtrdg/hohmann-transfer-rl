import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr

tbr = tbr()


def reward_exp(a):
    delta_a = tbr.r1*2 - a
    return np.exp(-(delta_a/((2-1.3)*tbr.r1))**2)

def reward_space(reward_func, tbr):
    a = np.linspace(tbr.r1*1.3, 2*tbr.r1, 25)
    r = reward_func(a)
    return a, r

a, r = reward_space(reward_exp, tbr)
r_min, r_max = -np.abs(r).max(), np.abs(r).max()

plt.plot(a, r, markersize=2)
plt.show()





