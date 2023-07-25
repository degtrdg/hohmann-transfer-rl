import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om
from SimpleBurnEnv import SimpleBurnEnv as sbe
from stable_baselines3 import PPO

env = sbe()
env.offset = np.pi/3
env.reset()

# Load model
model_dir = "models/simple/DQN"
model = PPO.load(f'{model_dir}/90000')

plt.figure(figsize=(8,5))
plt.subplot(2,3,1)
plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.title("Orbit")

# Initial orbit
trajectory = om.orbit_trajectory(env.orbit_state[1:3], env.orbit_state[3])
plt.plot(trajectory[0,:], trajectory[1,:], 'black', alpha=1)

t = 0
max_t = np.min([200, int(env.max_t/10)])
positions = np.empty((max_t,2))
orbits = np.empty((max_t, 4))
states = np.empty((max_t,6))
actions = np.empty(max_t)

while True:
    action, _states = model.predict(env.state)
    if env.step(action)[2] or t >= max_t:
        break
    positions[t,:] = env.ivp_state[0:2]
    states[t,:] = env.state
    orbits[t,:] = env.orbit_state
    actions[t] = 1
    t += 1

positions = positions[:t,:]
states = states[:t,:]
orbits = orbits[:t,:]
actions = actions[:t]

# Target orbit
trajectory = om.orbit_trajectory(env.target[0], env.target[1])
plt.plot(trajectory[0,:], trajectory[1,:], 'green', alpha=1)

# Trajectory of the agent
plt.subplot(2,3,1)
plt.plot(positions[:,0], positions[:,1], 'darkorange', alpha=1)
plt.plot(positions[-1,0], positions[-1,1], 'red', alpha=1, marker='x', markersize=5)

# Reward space
plt.subplot(2,3,2)
plt.title("Reward space")
def reward_space(reward_func):
    e_difference = np.linspace(0, 1, 25)
    a_difference = np.linspace(0, 1, 25)

    e, a = np.meshgrid(e_difference, a_difference)
    r = reward_func(e, a)
    r = r[:-1, :-1]
    return e, a, r

e, a, r = reward_space(lambda e,a: np.exp(-((e)**2 + a**2)))
r_min, r_max = -np.abs(r).max(), np.abs(r).max()

c = plt.pcolormesh(e, a, r, cmap='RdBu', vmin=r_min, vmax=r_max)

# Reward space trajectory
target_deltas = np.array([[np.linalg.norm(np.array([states[i,1], states[i,2]]))/np.linalg.norm(env.target[0]), 
                           (states[i,3])/(states[0,3])] for i in range(t)])
plt.scatter(target_deltas[:,0], target_deltas[:,1], color='k', marker='x', linewidth=.5, s=5)

# Reward plot
plt.subplot(2,3,3)
rewards = np.array([env.reward(states[i,:], actions[i]) for i in range(t)])
plt.title("Reward")
plt.plot(rewards, color='blue')
ax2 = plt.twinx()
ax2.plot(np.cumsum(rewards), color='orange')

# Target time
plt.subplot(2,3,4)
plt.title("Delta a")
plt.plot(states[:,3]/env.tbr.r1)

# Thrust remaining
plt.subplot(2,3,5)
plt.title("angle diff")
plt.plot(states[:,4])

# True anomaly
plt.subplot(2,3,6)
plt.title("target time")
plt.plot(states[:,5])

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

