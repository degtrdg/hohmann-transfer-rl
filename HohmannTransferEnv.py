import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dopri45 import DoPri45Step, DoPri45integrate

class HohmannTransferEnv(gym.Env):
    """
    Description:
        A spaceship is in an orbit and its goal is to reach a higher orbit using Hohmann transfer.
    Observation: 
        Type: Box(7)
        Num     Observation             Min    Max
        0       Spaceship x position    -Inf   Inf
        1       Spaceship y position    -Inf   Inf
        2       Spaceship x velocity    -Inf   Inf
        3       Spaceship y velocity    -Inf   Inf
        4       Gravitational x force   -Inf   Inf
        5       Gravitational y force   -Inf   Inf

    Actions:
        Type: Box(2)
        Num   Action
        0     Thrust
        1     Angle -- relative to negative velocity vector
        
    Note: The spaceship is considered a point mass.
    """

    def __init__(self):
        self.min_actions = np.array([0, -np.pi])
        self.max_actions = np.array([0.1, np.pi])
        self.min_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self.max_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Might need to scale these values
        self.gravity_constant = 6.67430e-11  # in m^3 kg^-1 s^-2
        self.earth_mass = 5.972e24  # in kg
        self.earth_radius = 6371000  # in m
        self.spaceship_mass = 1000  # in kg, just an assumption
        self.epsilon = 1e-6  # to avoid division by zero

        # Recieved warning about casting float64 to float32, flagging for later
        self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(6,), dtype=np.float64)

        self.state = None  # will be initialized in reset method

    def Fg(self, pos):
        # position needs to be passed in as a numpy array
        return -(self.gravity_constant * self.earth_mass * self.spaceship_mass / np.linalg.norm(pos)**3) * pos

    def orbit_velocity(self, pos):
        # position needs to be passed in as a numpy array
        speed = np.sqrt(self.gravity_constant * self.earth_mass / np.linalg.norm(pos))
        return np.array([pos[1], -pos[0]]) / np.linalg.norm(pos) * speed
    
    def ode(self, y, thrust):
        pos = y[:2]
        vel = y[2:]

        dydt = np.zeros(4)
        dydt[:2] = vel
        dydt[2:] = (self.Fg(pos) + thrust) / self.spaceship_mass

        return dydt

    def step(self, action, dt=0.01):
        # Implement the dynamics of the environment here.
        # Use scipy's solve_ivp to get the new state given the current state and action.
        # Remember to respect the constraints of the environment.

        # Limit the action space
        action = np.clip(action, self.min_actions, self.max_actions)

        # Seperate position, velocity, and force
        vel = np.array([self.state[2], self.state[3]])

        # Calculate the thrust vector
        if np.linalg.norm(vel) < self.epsilon:
            thrust = np.array([0, -action[0]])
        else:
            thrust = np.matmul(np.array([[np.cos(action[1]), -np.sin(action[1])], 
                                        [np.sin(action[1]), np.cos(action[1])]]), 
                                        -vel / np.linalg.norm(vel) * action[0])
        
        y0 = np.array(self.state[:4])

        # My attempt to fix step size in rk45, still slight performance defecit compared to euler, but far more accurate
        sol = solve_ivp(lambda t,y: self.ode(y, thrust), (0, dt), y0, method='RK45', t_eval=[dt], max_step=dt, atol = 1, rtol = 1)
        self.state[:4] = sol.y[:, 0]
        self.state[4:] = self.Fg(np.array(self.state[:2]))

        # TODO: Add terminal condition and reward
        return self.state, 1, False, {}
        

    def euler_step(self, action, dt=0.01):
        # Limit the action space
        action = np.clip(action, self.min_actions, self.max_actions)

        # Seperate position, velocity, and force
        pos = np.array([self.state[0], self.state[1]])
        vel = np.array([self.state[2], self.state[3]])
        Fg = np.array([self.state[4], self.state[5]])

        if np.linalg.norm(vel) < self.epsilon:
            thrust = np.array([0, action[0]])
        else:
            thrust = np.matmul(np.array([[np.cos(action[1]), -np.sin(action[1])], 
                                        [np.sin(action[1]), np.cos(action[1])]]), 
                                        -vel / np.linalg.norm(vel) * action[0])
         

        a = (Fg + thrust) / self.spaceship_mass
        vel_ = vel + a*dt
        pos_ = pos + vel*dt + 0.5*a*dt**2
        Fg_ = self.Fg(pos_)

        self.state = [pos_[0], pos_[1], vel_[0], vel_[1], Fg_[0], Fg_[1]]

        return self.state, 1, False, {}

    def reset(self, pos=None, vel=None, Fg=None):
        # Define the initial state here.
        # The state vector is: [x, y, vx, vy, Fgx, Fgy]
        if pos is None:
            pos = np.array([self.earth_radius * 1.3, 0])
        if vel is None:
            vel = self.orbit_velocity(pos)
        if Fg is None:
            Fg = self.Fg(pos)

        self.state = [pos[0], pos[1], vel[0], vel[1], Fg[0], Fg[1]]
        return self.state


    def render(self, mode='human'):
        pass


