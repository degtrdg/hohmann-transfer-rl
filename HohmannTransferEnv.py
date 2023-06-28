import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr

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
        4       Eccentricity x          -Inf   Inf
        5       Eccentricity y          -Inf   Inf
        6       Semi-major axis         -Inf   Inf

    Actions:
        Type: Box(2)
        Num   Action
        0     Thrust
        1     Angle
        
    Note: The spaceship is considered a reduced mass.
    """
    def __init__(self):
        # TODO: use a symmetric and normalized Box action space
        # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.min_actions = np.array([0, -np.pi])
        self.max_actions = np.array([1000, np.pi])
        self.min_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self.max_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Starting orbit
        self.tbr = tbr()
        self.r20 = self.tbr.r1*1.3
        self.v20 = self.tbr.circ_velocity(self.r20)
        self.e0 = om.eccentricity_vector(self.r20, self.v20, self.tbr.mu)
        self.a0 = om.semi_major_axis(self.r20, self.v20, self.tbr.mu)

        # Target orbit
        self.max_a = 4
        self.min_a = 2
        self.max_c = 2

        self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(7,), dtype=np.float64)

        self.state = None  # will be initialized in reset method

    def reward(self, state, target):
        e0 = np.array([state[4], state[5]])
        delta_e = np.linalg.norm(e0 - target[0])

        a0 = state[6]
        delta_a = target[1] - a0
        return np.exp(-(delta_a/((self.max_a*tbr.r1-self.a0)))**2) * np.exp(-(delta_e/(self.max_c/self.max_c))**2)

    def step(self, action, dt=0.01):
        # Limit the action space
        action = np.clip(action, self.min_actions, self.max_actions)
        vel = np.array([self.state[2], self.state[3]])

        # Calculate the thrust vector
        if np.linalg.norm(vel) < self.tbr.epsilon:
            thrust = np.array([0, -action[0]])
        else:
            thrust = np.matmul(np.array([[np.cos(action[1]), -np.sin(action[1])], 
                                        [np.sin(action[1]), np.cos(action[1])]]), 
                                        -vel / np.linalg.norm(vel) * action[0])
        
        # Initial conditions for the ODE solver
        y0 = np.array(self.state[:4])

        # My attempt to fix step size in rk45, still slight performance defecit compared to euler, but far more accurate
        sol = solve_ivp(lambda t,y: self.tbr.ode(y, thrust), (0, dt), y0, method='RK45', t_eval=[dt], max_step=dt, atol = 1, rtol = 1)

        # Update the state
        self.state[:4] = sol.y[:, 0]
        self.state[4:6] = om.eccentricity_vector(self.state[:2], self.state[2:4], self.tbr.mu)
        self.state[6] = om.semi_major_axis(self.state[:2], self.state[2:4], self.tbr.mu)

        # TODO: Add further terminal conditions and reward
        reward = self.reward(self.state, self.target)
        terminal = False
        
        if np.linalg.norm(self.state[:2]) <= self.tbr.r1:
            terminal = True
        
        # TODO: add truncated var for if it crashes, goes out of bounds, etc.
        truncated = False

        info =  {}

        return self.state, reward, terminal, truncated, info

    def reset(self, seed=None, options=None, pos=None, vel=None, e=None, a=None):
        # Define the initial state here.
        # The state vector is: [x, y, vx, vy, Fgx, Fgy]
        if pos is None:
            pos = self.r20
        if vel is None:
            vel = self.v20
        if e is None:
            e = self.e0
        if a is None:
            a = self.a0

        self.state = [pos[0], pos[1], vel[0], vel[1], e[0], e[1], a]
        self.target = om.random_orbit(self.tbr, max_a=self.max_a, min_a=self.min_a, max_c=self.max_c)
        info = {}
        return self.state, info

    def render(self, mode='human'):
        pass
