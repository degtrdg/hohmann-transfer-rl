import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import orbital_mechanics as om

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
        
    Note: The spaceship is considered a point mass.
    """

    def __init__(self):
        self.min_actions = np.array([0, -np.pi])
        self.max_actions = np.array([1000, np.pi])
        self.min_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self.max_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Might need to scale these values
        self.gravity_constant = 6.67430e-11  # in m^3 kg^-1 s^-2
        self.earth_mass = 5.972e24  # in kg
        self.mu = self.gravity_constant * self.earth_mass
        self.earth_radius = 6371000  # in m
        self.spaceship_mass = 1000  # in kg, just an assumption
        self.epsilon = 1e-6  # to avoid division by zero

        # Recieved warning about casting float64 to float32, flagging for later
        self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(7,), dtype=np.float64)

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
        vel = np.array([self.state[2], self.state[3]])

        # Calculate the thrust vector
        if np.linalg.norm(vel) < self.epsilon:
            thrust = np.array([0, -action[0]])
        else:
            thrust = np.matmul(np.array([[np.cos(action[1]), -np.sin(action[1])], 
                                        [np.sin(action[1]), np.cos(action[1])]]), 
                                        -vel / np.linalg.norm(vel) * action[0])
        
        # Initial conditions for the ODE solver
        y0 = np.array(self.state[:4])

        # My attempt to fix step size in rk45, still slight performance defecit compared to euler, but far more accurate
        sol = solve_ivp(lambda t,y: self.ode(y, thrust), (0, dt), y0, method='RK45', t_eval=[dt], max_step=dt, atol = 1, rtol = 1)

        # Update the state
        self.state[:4] = sol.y[:, 0]
        self.state[4:6] = om.eccentricity_vector(self.state[:2], self.state[2:4], self.mu)
        self.state[6] = om.semi_major_axis(self.state[:2], self.state[2:4], self.mu)

        # TODO: Add further terminal conditions and reward
        reward = 0
        terminal = False
        
        if np.linalg.norm(self.state[:2]) <= self.earth_radius:
            terminal = True

        return self.state, reward, terminal, {}

    def reset(self, pos=None, vel=None, e=None, a=None):
        # Define the initial state here.
        # The state vector is: [x, y, vx, vy, Fgx, Fgy]
        if pos is None:
            pos = np.array([self.earth_radius * 1.3, 0])
        if vel is None:
            vel = self.orbit_velocity(pos)
        if e is None:
            e = om.eccentricity_vector(pos, vel, self.mu)
        if a is None:
            a = om.semi_major_axis(pos, vel, self.mu)

        self.state = [pos[0], pos[1], vel[0], vel[1], e[0], e[1], a]
        return self.state

    def render(self, mode='human'):
        pass


