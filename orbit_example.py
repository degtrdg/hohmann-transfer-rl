import HohmannTransferEnv
import numpy as np
import matplotlib.pyplot as plt

env = HohmannTransferEnv.HohmannTransferEnv()
env.reset()
positions_euler = {}
positions_rk45 = {}
velocities = [np.array(env.state[3]) * i for i in [1.1]]
t = 3000
dt = 5

earth = plt.Circle((0, 0), env.earth_radius, facecolor='none', edgecolor='black', linestyle='--')
fig, ax = plt.subplots()
ax.add_patch(earth)

print("Calculating trajectories...")
for v in velocities:
    print(v)
    env.reset(vel=np.array([0, v]))
    positions_rk45[v] = np.empty((t, 2))
    for i in range(t):
        positions_rk45[v][i] = env.state[:2]
        env.step(np.array([0, 0]), dt=dt)
        if np.linalg.norm(env.state[:2]) < env.earth_radius:
            positions_rk45[v] = positions_rk45[v][:i]
            break
    print("*")
    env.reset(vel=np.array([0, v]))
    positions_euler[v] = np.empty((t, 2))
    for i in range(t):
        positions_euler[v][i] = env.state[:2]
        env.euler_step(np.array([0, 0]), dt=dt)
        if np.linalg.norm(env.state[:2]) < env.earth_radius:
            positions_euler[v] = positions_euler[v][:i]
            break
    

for v in velocities:
    plt.plot(positions_euler[v][:, 0], positions_euler[v][:, 1])
    plt.plot(positions_rk45[v][:, 0], positions_rk45[v][:, 1])
plt.show()
