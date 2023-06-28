import numpy as np

class TwoBodyReduced():
    def __init__(self, gravity_constant=6.67430e-11, m1=5.972e24, r1=6371000, m2=1000, epsilon=1e-5):
        self.gravity_constant = gravity_constant
        self.m1 = m1
        self.m2 = m2
        self.mu = gravity_constant * m1
        self.r1 = r1
        self.epsilon = epsilon

    def Fg(self, pos):
        return -(self.mu * self.m2 / np.linalg.norm(pos)**3) * pos
    
    def ode(self, y, thrust):
        pos = y[:2]
        vel = y[2:]

        dydt = np.zeros(4)
        dydt[:2] = vel
        dydt[2:] = (self.Fg(pos) + thrust) / self.m2

        return dydt
    
    def circ_velocity(self, pos):
        speed = np.sqrt(self.gravity_constant * self.m1 / np.linalg.norm(pos))
        return np.array([pos[1], -pos[0]]) / np.linalg.norm(pos) * speed
    

