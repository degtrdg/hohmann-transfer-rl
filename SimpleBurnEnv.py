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
        0       True anomaly            0      2*pi
        1       Delta eccentricity x    -1     1
        2       Delta eccentricity y    -1     1
        3       Delta semi-major axis   -Inf   Inf
        4       Thrust remaining        0      150
        5       Previous action         0      1

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Thrust (0 or 1)
    
    Explanations:
    State space: 
        - True anomaly: The angle from the direction of periapsis measured in the direction of motion.
        - Eccentricity x: The x component of the eccentricity vector (direction towards periapsis).
        - Eccentricity y: The y component of the eccentricity vector (direction towards periapsis).
        - Semi-major axis: One half the major axis, and thus runs from the center of the body, 
          through a focus, and to the perimeter of the orbit.
        - Time to apoapsis: The time remaining until the spaceship reaches the apoapsis (highest point in the orbit).
        - Thrust remaining: The remaining amount of thrust.

    Action space: 
        - Thrust (0 or 1): Thrust can either be off (0) or on (1).
        
    Note: The spaceship is considered a reduced mass.
    """
    def __init__(self):
        # TODO: use a symmetric and normalized Box action space
        # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

        # Maximum allowable magnitude of an action, currently set to 1000.
        self.max_action = 1000

        # The minimum and maximum values for each observation parameter, used to define the bounds of the observation space.
        # They are arrays of 6 values, corresponding to the six parameters outlined in the 'Observation' section.
        self.min_obs = np.array([0, -1, -1, -np.inf, 0, 0, 0])
        self.max_obs = np.array([2*np.pi, 1, 1, np.inf, 150, 1, np.inf])

        # The maximum time for the simulation. After this time has passed, the simulation will end.
        self.max_t = 10000 

        # The starting time for the simulation, set to 0 for the start of the simulation.
        self.t0 = 0 

        # The action space and observation space for the simulation.
        # The action space is discrete and can be either 0 or 1, corresponding to no thrust or full thrust.
        # The observation space is continuous and is defined by the min_obs and max_obs arrays.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(7,), dtype=np.float64)

        # This is a reference to an object representing the two-body problem. It provides necessary functions for orbit calculation.
        self.tbr = tbr()

        # Initial position (r2_0) and velocity (v2_0) vectors for the spaceship. The "2" indicates the spaceship as the second body in the system.
        self.r2_0 = np.array([self.tbr.r1*1.3,0])
        self.v2_0 = self.tbr.circ_velocity(self.r2_0)

        # The orbital elements at the start of the simulation, calculated from the initial position and velocity.
        # nu0 is the true anomaly, e0 is the eccentricity vector, a0 is the semi-major axis, and ta is the time to apoapsis.
        self.nu0 = om.true_anomaly(self.r2_0, self.v2_0, self.tbr.mu)
        self.e0 = om.eccentricity_vector(self.r2_0, self.v2_0, self.tbr.mu)
        self.a0 = om.semi_major_axis(self.r2_0, self.v2_0, self.tbr.mu)
        # The initial amount of thrust available to the spaceship.
        self.thrusts = 150

        # The semi-major axis of the target orbit.
        self.target_a = 2

        # The initial values of the integrator state and the environment state, set to None as they are initialized in the reset method.
        self.ivp_state = None
        self.orbit_state = None
        self.state = None
        self.target = None

    def reward(self, state, action):
        delta_e = np.linalg.norm(state[1:3])/np.linalg.norm(self.target[0])
        delta_a = state[3]/(self.target[1]-self.a0)

        if action == state[5]:
            continuity_reward = 0.1
        else:
            continuity_reward = 0
        
        return np.exp(-((delta_e)**2)) + continuity_reward

    def step(self, action, dt=10):
        # TODO: Add further terminal conditions and reward

        # The step function is called at each time step of the simulation. It advances the state of the spaceship 
        # by applying the given action and updating the spaceship's position and velocity accordingly.

        self.t0 += dt  # Advance the current time by the time step dt.

        # Extract the velocity vector from the current state.
        vel = np.array([self.ivp_state[2], self.ivp_state[3]])

        # Calculate the thrust vector. The direction of the thrust is the same as the direction of the velocity
        # (i.e., the spaceship thrusts in the direction it is currently moving). The magnitude of the thrust is
        # determined by the action times the maximum thrust. The action is either 0 (no thrust) or 1 (maximum thrust).
        thrust = vel / np.linalg.norm(vel) * action * self.max_action
        
        # Initial conditions for the ODE (Ordinary Differential Equation) solver, which we will use to
        # integrate the spaceship's motion over the time step.
        y0 = np.array(self.ivp_state)

        # We solve the motion ODE for the given thrust over the time interval [0, dt], starting from the initial conditions y0.
        # The solution gives us the state of the spaceship (position and velocity) at the end of the time step.
        sol = solve_ivp(lambda t,y: self.tbr.ode(y, thrust), (0, dt), y0, method='RK45', t_eval=[dt], max_step=dt, atol = 1, rtol = 1)

        # Update the state with the new position and velocity from the ODE solution.
        self.ivp_state = sol.y[:, -1]
        # Calculate and update various orbital parameters based on the new state.
        self.orbit_state[0] = om.target_relative_anomaly(self.ivp_state[:2], self.target)
        if self.orbit_state[0] > np.pi:
            self.orbit_state[0] -= 2*np.pi
        self.orbit_state[1:3] = om.eccentricity_vector(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)
        self.orbit_state[3] = om.semi_major_axis(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)

        self.state[0] = self.orbit_state[0]
        self.state[1:3] = self.target[0] - self.orbit_state[1:3]
        self.state[3] = self.target[1] - self.orbit_state[3]
        # If the action was to thrust, decrease the amount of remaining thrust.
        if action != 0:
            self.state[4] -= 1
        self.state[5] = action
        # self.state[6] = om.absolute_target_time(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu, self.target)
        self.state[6] = om.absolute_target_time(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu, self.target)
        # self.state[6] = om.time_at_state(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu)

        # Calculate the reward for the current state.
        reward = self.reward(self.state, action)
        e_norm = np.linalg.norm(self.orbit_state[1:3])

        # Check termination conditions: either the maximum time has been reached, or the spaceship has achieved its goal,
        # or the spaceship has run out of thrust, or the spaceship's orbit is too eccentric. If any of these conditions is met,
        # the simulation is terminated.
        if self.t0 >= self.max_t or e_norm >= .6:
            truncated = True
        else:
            truncated = False

        terminal = truncated

        # Return the new state, the reward, and the termination status. The 'info' dictionary can be used to provide additional
        # information about the state of the simulation, but in this case it is empty.
        info =  {}
        return self.state, reward, terminal, truncated, info

    def reset(self, seed=None, options=None, theta=5*np.pi/4, thrusts=None, target=None):
        # The reset function is called to reset the environment to its initial state.
        self.t0 = 0  # Reset the current time to 0.

        # Set the initial position and velocity of the spaceship.
        pos = self.r2_0
        vel = self.v2_0

        if target is None:
            self.target = om.a_constrained_orbit(self.tbr, r=np.linalg.norm(pos)/self.tbr.r1, a=self.target_a, theta=theta)
        else:
            self.target = target
        
        # Calculate various orbital parameters based on the initial position and velocity.
        nu = om.target_relative_anomaly(pos, self.target)
        if nu > np.pi:
            nu -= 2*np.pi
        e = om.eccentricity_vector(pos, vel, self.tbr.mu)
        a = om.semi_major_axis(pos, vel, self.tbr.mu)
        
        # Check if a specific amount of thrust has been specified for the reset; if not, use the default amount.
        if thrusts is None:
            thrusts = self.thrusts

        # Set the initial state for the ODE solver.
        self.ivp_state = np.array([pos[0], pos[1], vel[0], vel[1]])
        # Set the initial state for the simulation.
        self.orbit_state = np.array([nu, e[0], e[1], a])
        # Set the target orbit for the simulation.
        self.state = np.array([nu, self.target[0][0]-e[0], self.target[0][1]-e[1], 
                               self.target[1]-a, thrusts, 0, 
                               om.absolute_target_time(self.ivp_state[:2], self.ivp_state[2:4], self.tbr.mu, self.target)])
        
        info = {}  # The info dictionary can be used to provide additional information about the state of the simulation, but in this case it is empty.

        # Return the initial state and the info dictionary.
        return self.state, info

    def render(self, mode='human'):
        pass
