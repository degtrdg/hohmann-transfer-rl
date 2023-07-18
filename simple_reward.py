import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from SimpleBurnEnv import SimpleBurnEnv as sbe

env = sbe()
env.reset()

plt.figure(figsize=(8,5))
plt.subplot(2,3,1)
plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.title("Orbit")
# Initial orbit
trajectory = om.orbit_trajectory(env.orbit_state[1:3], env.orbit_state[3])
plt.plot(trajectory[0,:], trajectory[1,:], 'black', alpha=1)

t = 0
burn_start = 100
max_t = 1700
positions = np.empty((max_t,2))
orbits = np.empty((max_t, 4))
states = np.empty((max_t,6))
actions = np.empty(max_t)

while True:
    if t >= burn_start:
        env.step(1)
    else:
        env.step(0)
    if env.state[3] <= 0:
        break
    t += 1
t = 0
target = (env.orbit_state[1:3], env.orbit_state[3])
print("Setting target orbit:", target[0], np.linalg.norm(target[0]), target[1]/env.tbr.r1)
env.reset(target=target)
print("Recieved target:", env.target[0], np.linalg.norm(env.target[0]), env.target[1]/env.tbr.r1)
print("Starting simulation...\n")

while True:
    if t >= burn_start:
        env.step(1)
    else:
        env.step(0)
    if env.state[3] <= 0:
        for i in range(t, max_t):
            env.step(0)
            positions[i,:] = env.ivp_state[0:2]
            states[i,:] = env.state
            orbits[i,:] = env.orbit_state
            actions[i] = 0
        break
    positions[t,:] = env.ivp_state[0:2]
    states[t,:] = env.state
    orbits[t,:] = env.orbit_state
    actions[t] = 1
    t += 1

# Target orbit
trajectory = om.orbit_trajectory(np.array([env.orbit_state[1], env.orbit_state[2]]), env.orbit_state[3])
plt.plot(trajectory[0,:], trajectory[1,:], 'green', alpha=1)

# Trajectory of the agent
plt.plot(positions[:t,0], positions[:t,1], 'darkorange', alpha=1)
plt.plot(positions[-1,0], positions[-1,1], 'blue', alpha=1, marker='x', markersize=5)

# Reward space
plt.subplot(2,3,2)
plt.title("Reward space")
def reward_space(reward_func, env, target):
    e_difference = np.linspace(0, 1, 25)
    a_difference = np.linspace(0, 1, 25)

    e, a = np.meshgrid(e_difference, a_difference)
    r = reward_func(e, a)
    r = r[:-1, :-1]
    return e, a, r

e, a, r = reward_space(lambda e,a: np.exp(-(e**2 + a**2)), env, target)
r_min, r_max = -np.abs(r).max(), np.abs(r).max()

c = plt.pcolormesh(e, a, r, cmap='RdBu', vmin=r_min, vmax=r_max)

# Reward space trajectory
target_deltas = np.array([[np.linalg.norm(np.array([states[i,1], states[i,2]]))/np.linalg.norm(target[0]), 
                           (states[i,3])/(target[1] - orbits[0,3])] for i in range(t)])
plt.scatter(target_deltas[:,0], target_deltas[:,1], color='k', marker='x', linewidth=.5, s=5)

# Reward plot
plt.subplot(2,3,3)
rewards = np.array([env.reward(states[i,:], actions[i]) for i in range(max_t)])
plt.title("Reward")
plt.plot(rewards, color='blue')
ax2 = plt.twinx()
ax2.plot(np.cumsum(rewards), color='orange')

# Thrust remaining
plt.subplot(2,3,5)
plt.title("Thrust Remaining")
plt.plot(states[:,4])

# True anomaly
plt.subplot(2,3,6)
plt.title("Relative True Anomaly")
plt.plot(states[:,0])

# Info about the orbit
print("total steps:", len(positions))
print("a accuracy (target, error):", 2, env.state[3]/env.tbr.r1)
print("e accuracy (target, result):", np.linalg.norm(env.target[0]), np.sqrt(env.orbit_state[1]**2 + env.orbit_state[2]**2))
print("--------------------")
print("orbit state:", env.orbit_state[1:3], np.linalg.norm(env.orbit_state[1:3]), env.orbit_state[3]/env.tbr.r1)
print("state:", env.state[1:3], np.linalg.norm(env.state[1:3]), env.state[3]/env.tbr.r1)
print("target", env.target[0], np.linalg.norm(env.target[0]), env.target[1]/env.tbr.r1)
print("--------------------")
print("reward:", env.reward(env.state, 0))
print("total reward:", np.sum(rewards))
print("initial reward:", rewards[0])
plt.show()

