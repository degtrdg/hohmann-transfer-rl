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

        # Limit the action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Define the ODE
        def ode(t, y):
            # What do each of the elements of y represent?
            # y[0] = x
            # y[1] = y
            # y[2] = vx
            # y[3] = vy
            # y[4] = Fg
            r = np.sqrt(y[0]**2 + y[1]**2)
            return [y[2],  # dx/dt = vx
                    y[3],  # dy/dt = vy
                    -self.gravity_constant * self.earth_mass / r**3 * y[0] + action[0]/self.spaceship_mass,  # dvx/dt = ax
                    -self.gravity_constant * self.earth_mass / r**3 * y[1] + action[1]/self.spaceship_mass,  # dvy/dt = ay
                    self.gravity_constant * self.earth_mass / r**2]  # dFg/dt = 0

        # Define the time span of the integration
        t_span = [0, 1]  # 1-second step

        # Solve the ODE
        sol = solve_ivp(ode, t_span, self.state)

        # The new state is the final value of the solution
        next_state = sol.y[:, -1]

        # Define the reward function
        reward = 1  # placeholder for the reward function

        # Define the terminal condition
        r = np.sqrt(next_state[0]**2 + next_state[1]**2)
        done = r <= 10 or r >= 100  # The episode ends if the spaceship is within 10 units from the center of the Earth or more than 100 units away.

        self.state = next_state
        return next_state, reward, done, {}

    def reset(self):
        # Define the initial state here.
        # The state vector is: [x, y, vx, vy, Fg]
        self.state = np.array([0., 0., 0., 0., self.gravity_constant * self.earth_mass])
        return self.state


    def render(self, mode='human'):
        pass
