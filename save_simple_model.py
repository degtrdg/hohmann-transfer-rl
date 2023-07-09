from SimpleBurnEnv import SimpleBurnEnv as sbe
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import numpy as np
import matplotlib.pyplot as plt
import orbital_mechanics as om


# Directories
model_dir = "models/simple/DQN"
logdir = 'logs'

# Make directories if they don't exist 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = sbe()
# reset the environment
env.reset()
# After we create the env and reset we can add the model with the algo
# model = DQN('MlpPolicy', env, verbose=1, tau=0.9)
model = PPO('MlpPolicy', env, verbose=1)

TIMESTAMPS = 5000
# Train the model and save it
for i in range(1,101):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, progress_bar=False)
    model.save(path=f'{model_dir}/{TIMESTAMPS*i}')
    
