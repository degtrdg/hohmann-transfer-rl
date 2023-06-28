import HohmannTransferEnv
import gymnasium as gym
from stable_baselines3 import PPO
import os


# Directories
model_dir = "models/PPO"
logdir = 'logs'

# Make directories if they don't exist 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = HohmannTransferEnv.HohmannTransferEnv()

# reset the environment
env.reset()

# After we create the env and reset we can add the model with the algo
# Add tensorboard
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTAMPS = 1000
# Train the model and save it

for i in range(1,30):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(path=f'{model_dir}/{TIMESTAMPS*i}')
