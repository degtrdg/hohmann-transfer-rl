import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import orbital_mechanics as om
from TwoBodyReduced import TwoBodyReduced as tbr

class SimpleBurnEnv(gym.Env):
    """
    Description:
        A spaceship is in an orbit and its goal is to reach a higher orbit using Hohmann transfer.
    Observation: 
        Type: Box(7)
        Num     Observation             Min    Max
        0       True anomaly            -pi    pi
        1       Eccentricity x          -Inf   Inf
        2       Eccentricity y          -Inf   Inf
        3       Semi-major axis         -Inf   Inf
        
        4       Time to apoapsis        0      Inf
        5       Thrust remaining        0      150

        6       Target a                -Inf   Inf
        7       Target e x              -Inf   Inf
        8       Target e y              -Inf   Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Thrust (0 or 1)
        
    Note: The spaceship is considered a reduced mass.
    """
    def __init__(self):
        # TODO: use a symmetric and normalized Box action space
        # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        self.max_action = 1000
        self.min_obs = np.array([0, -np.inf, -np.inf, -np.inf, 0, 0])
        self.max_obs = np.array([2*np.pi, np.inf, np.inf, np.inf, np.inf, 150])
        
        self.max_t = 10000
        self.t0 = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(6,), dtype=np.float64)

        self.tbr = tbr()

        # Starting orbit
        self.r20 = np.array([self.tbr.r1*1.3,0])
        self.v20 = self.tbr.circ_velocity(self.r20)

        # Orbital elements
        self.nu0 = om.true_anomaly(self.r20, self.v20, self.tbr.mu)
        self.e0 = om.eccentricity_vector(self.r20, self.v20, self.tbr.mu)
        self.a0 = om.semi_major_axis(self.r20, self.v20, self.tbr.mu)
        self.ta = om.time_to_apoapsis(self.r20, self.v20, self.tbr.mu)
        self.thrusts = 150

        # Target orbit
        self.target_a = 2

        self.ivp_state = None  # will be initialized in reset method
        self.state = None  # will be initialized in reset method

    def reward(self, state, target):
        e0 = np.array([state[1], state[2]])
        delta_e = np.linalg.norm(target[0] - e0)

        a0 = state[3]
        delta_a = target[1] - a0
        return np.exp(-(2*delta_a/((target[1]-self.a0)))**2) * np.exp(-(delta_e/(self.target[1])**2))

    def step(self, action, dt=10):
        self.t0 += dt
        vel = np.array([self.ivp_state[2], self.ivp_state[3]])

        # Calculate the thrust vector
        thrust = vel / np.linalg.norm(vel) * action * self.max_action
        
        # Initial conditions for the ODE solver
        y0 = np.array(self.ivp_state)

        # My attempt to fix step size in rk45, still slight performance defecit compared to euler, but far more accurate
        sol = solve_ivp(lambda t,y: self.tbr.ode(y, thrust), (0, dt), y0, method='RK45', t_eval=[dt], max_step=dt, atol = 1, rtol = 1)

        # Update the state
        self.ivp_state = sol.y[:, -1]
        self.state[0] = om.true_anomaly(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)
        self.state[1:3] = om.eccentricity_vector(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)
        self.state[3] = om.semi_major_axis(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)
        self.state[4] = om.time_to_apoapsis(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)
        if action != 0:
            self.state[5] -= 1

        # TODO: Add further terminal conditions and reward
        reward = self.reward(self.state, self.target) - action
        e_norm = np.linalg.norm(self.state[1:3])

        if self.t0 >= self.max_t or self.state[3] >= (self.target_a+1)*self.tbr.r1 or e_norm >= .5 or self.state[5] <= 0:
            truncated = True
        else:
            truncated = False

        terminal = truncated
        
        info =  {}
        return self.state, reward, terminal, truncated, info

    def reset(self, seed=None, options=None, theta=0, thrusts=None):
        self.t0 = 0
        pos = self.r20
        vel = self.v20
        
        nu = om.true_anomaly(pos, vel, self.tbr.mu)
        e = om.eccentricity_vector(pos, vel, self.tbr.mu)
        a = om.semi_major_axis(pos, vel, self.tbr.mu)
        ta = om.time_to_apoapsis(pos, vel, self.tbr.mu)
        if thrusts is None:
            thrusts = self.thrusts

        self.ivp_state = np.array([pos[0], pos[1], vel[0], vel[1]])
        self.state = np.array([nu, e[0], e[1], a, ta, thrusts])
        self.target = om.a_constrained_orbit(self.tbr, r=np.linalg.norm(pos)/self.tbr.r1, a=self.target_a, theta=theta)
        info = {}
        return self.state, info

    def render(self, mode='human'):
        pass
