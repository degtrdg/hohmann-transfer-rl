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
        0       Spaceship position      -Inf   Inf
        1       Spaceship velocity      -Inf   Inf
        2       Gravitational force     0      Inf

    Actions:
        Type: Box(2)
        Num   Action
        0     Thrust vector
        
    Note: The spaceship is considered a point mass.
    """

    def __init__(self):
        self.min_action = -0.1
        self.max_action = 0.1
        self.gravity_constant = 6.67430e-11  # in m^3 kg^-1 s^-2
        self.earth_mass = 5.972e24  # in kg
        self.spaceship_mass = 1000  # in kg, just an assumption

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.state = None  # will be initialized in reset method

    def step(self, action):
        # Implement the dynamics of the environment here.
        # Use scipy's solve_ivp to get the new state given the current state and action.
        # Remember to respect the constraints of the environment.

        # Limit the action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Seperate position, velocity, and force
        pos = self.state[0]
        vel = self.state[1]
        Fg = self.state[2]

        a = Fg / self.spaceship_mass + action

        t = 0.1
        vel_ = vel + a*t
        pos_ = pos + vel*t + 0.5*a*t**2
        Fg_ = self.gravity_constant * self.earth_mass * self.spaceship_mass / np.linalg.norm(pos_)**2

        self.state = np.array([pos_, vel_, Fg_])

        return self.state, 1, False, {}

        # Define the ODE
        # def ode(t, y):
        #     dy, v = y
        #     Fg = self.gravity_constant * self.earth_mass * self.spaceship_mass / np.linalg.norm(dy)**2
        #     return [v, Fg * dy / (np.linalg.norm(dy)*self.spaceship_mass) + action]

        # # Define the time span of the integration
        # t_span = [0, 1]  # 1-second step

        # # Solve the ODE
        # sol = solve_ivp(ode, t_span, self.state)

        # # The new state is the final value of the solution
        # next_state = sol.y[:, -1]

        # # Define the reward function
        # reward = 1  # placeholder for the reward function

        # # Define the terminal condition
        # r = np.sqrt(next_state[0]**2 + next_state[1]**2)
        # done = r <= 10 or r >= 100  # The episode ends if the spaceship is within 10 units from the center of the Earth or more than 100 units away.

        # self.state = next_state
        # return next_state, reward, done, {}

    def reset(self):
        # Define the initial state here.
        # The state vector is: [[x, y], [vx, vy], Fg]
        self.state = np.array([[0,0], [0,0], [0,0]])
        return self.state


    def render(self, mode='human'):
        pass
