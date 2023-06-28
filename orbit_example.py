import HohmannTransferEnv
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from tqdm import tqdm

env = HohmannTransferEnv.HohmannTransferEnv()
env.reset()

positions_rk45 = {}
velocities = [np.array(env.state[3]) * i for i in [0.92, 1, 1.05, 1.1]]
apsis = {}

# product of 10,000 is a good time to run for
# decrease t to increase computation speed
t = 2000
dt = 5

earth = plt.Circle((0, 0), env.tbr.r1, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)

print("Simulating trajectories...")
for v in tqdm(velocities):
    env.reset(vel=np.array([0, v]))
    apsis[v] = om.apsis(env.state[:2], env.state[2:4], env.tbr.mu)
    positions_rk45[v] = np.empty((t, 2))
    for i in tqdm(range(t), leave=False):
        positions_rk45[v][i] = env.state[:2]
        env.step(np.array([0, 0]), dt=dt)
        if np.linalg.norm(env.state[:2]) < env.tbr.r1:
            positions_rk45[v] = positions_rk45[v][:i]
            break

s = env.state
cm = plt.get_cmap('gist_rainbow')
for (i,v) in enumerate(velocities):
    plt.plot(positions_rk45[v][:, 0], positions_rk45[v][:, 1], color=cm(1.*i/len(velocities)), label="v = " + str(v))
    plt.scatter(apsis[v][0][0], apsis[v][0][1], color=cm(1.*i/len(velocities)), marker='x')
    plt.scatter(apsis[v][1][0], apsis[v][1][1], color=cm(1.*i/len(velocities)), marker='o')
plt.grid(color='lightgray',linestyle='--')
plt.show()
