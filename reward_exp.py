import HohmannTransferEnv as hte
import numpy as np
import matplotlib.pyplot as plt

env = hte.HohmannTransferEnv()
env.reset()

print(env.state)
env.step(np.array([0, 0]))
print(env.state)
