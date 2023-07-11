import numpy as np

class TwoBodyReduced():
    """
    TwoBodyReduced is a class designed to model the dynamics of a two-body system 
    using the reduced mass and the position of the second object relative to the first.

    Attributes:
    -------------
    gravity_constant : float
        Universal gravitational constant. Its default value is set to the standard in SI units (6.67430e-11 m^3 kg^-1 s^-2).

    m1 : float
        Mass of the first body. Default value represents Earth's mass in kg.

    m2 : float
        Mass of the second body, which is smaller compared to m1. Default value is set to 1000 kg.

    mu : float
        Standard gravitational parameter, product of the gravitational constant and the mass of the first body.

    r1 : float
        Radius of the first body. Default value represents Earth's radius in meters.

    epsilon : float
        Small constant added to prevent division by zero, if needed.

    Methods:
    -------------
    Fg(pos)
        Calculates gravitational force acting on the second body.

    ode(y, thrust)
        Returns the ordinary differential equations for position and velocity of the second body.

    circ_velocity(pos)
        Calculates the circular velocity for a given position in orbit around the first body.
    """
    
    def __init__(self, gravity_constant=6.67430e-11, m1=5.972e24, r1=6371000, m2=1000, epsilon=1e-5):
        self.gravity_constant = gravity_constant
        self.m1 = m1
        self.m2 = m2
        self.mu = gravity_constant * m1 
        self.r1 = r1
        self.epsilon = epsilon

    def Fg(self, pos):
        # This method calculates the gravitational force that the second body experiences.
        # The formula used is derived from Newton's law of universal gravitation:
        # F = G * (m1 * m2 / r^2)
        # However, here the direction of the force is also important. The force is
        # towards the center of the first body, which is at the origin of the coordinate system.
        # Hence, we need to divide the force by the distance to get the direction,
        # and multiply it by the position vector.
        return -(self.mu * self.m2 / np.linalg.norm(pos)**3) * pos

    def ode(self, y, thrust):
        # This method formulates the ordinary differential equations (ODEs) that describe 
        # the motion of the second body in the gravitational field of the first body.
        # The state of the system can be described by a 4D vector, y, 
        # with the first two elements being the position of the second body
        # and the last two elements being the velocity.
        # The derivative of this state vector, dydt, 
        # is a 4D vector where the first two elements are the velocity of the body 
        # (which is the derivative of the position) and the last two elements are the acceleration 
        # (which is the derivative of the velocity), 
        # given by the sum of gravitational force and any additional thrust divided by the mass of the second body.

        # Separate position and velocity from the input y
        pos = y[:2]
        vel = y[2:]
        # Initialize the array to store the derivatives
        dydt = np.zeros(4)
        # First two elements of dydt are velocity
        dydt[:2] = vel
        # Last two elements are acceleration, calculated as total force (gravity + thrust) divided by mass
        dydt[2:] = (self.Fg(pos) + thrust) / self.m2

        return dydt

    def circ_velocity(self, pos):
        # This method calculates the magnitude of the velocity that the second body must have 
        # in order to maintain a stable circular orbit around the first body at a given position.
        # According to the physics of circular orbits, this velocity v is given by 
        # v = sqrt(G * m1 / r), where r is the distance from the center of the first body.
        # The direction of this velocity is always perpendicular to the radial direction,
        # hence the term [pos[1], -pos[0]], which gives a vector that is rotated 90 degrees
        # counterclockwise from the position vector. 
        speed = np.sqrt(self.gravity_constant * self.m1 / np.linalg.norm(pos))
        return np.array([pos[1], -pos[0]]) / np.linalg.norm(pos) * speed
