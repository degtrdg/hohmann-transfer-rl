import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp

class HohmannTransferEnv(gym.Env):
    """
    Description:
        A spaceship is in an orbit and its goal is to reach a higher orbit using Hohmann transfer.
    Observation: 
        Type: Box(5)
        Num     Observation             Min    Max
        0       Spaceship position x    -Inf   Inf
        1       Spaceship position y    -Inf   Inf
        2       Spaceship velocity x    -Inf   Inf
        3       Spaceship velocity y    -Inf   Inf
        4       Gravitational force     0      Inf

    Actions:
        Type: Box(2)
        Num   Action
        0     Thrust in x direction
        1     Thrust in y direction
        
    Note: The spaceship is considered a point mass.
    """

    def __init__(self):
        self.min_action = -0.1
        self.max_action = 0.1
        self.gravity_constant = 6.67430e-11  # in m^3 kg^-1 s^-2
        self.earth_mass = 5.972e24  # in kg
        self.spaceship_mass = 1000  # in kg, just an assumption

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.state = None  # will be initialized in reset method

    def step(self, action):
        # Implement the dynamics of the environment here.
        # Use scipy's solve_ivp to get the new state given the current state and action.
        # Remember to respect the constraints of the environment.
        pass

    def reset(self):
        # Define the initial state here.
        # The state vector is: [x, y, vx, vy, Fg]
        pass

    def render(self, mode='human'):
        pass
