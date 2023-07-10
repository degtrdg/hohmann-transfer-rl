import numpy as np


# Orbital elements calculations
def angular_momentum(pos, vel):
    return np.cross(pos, vel)

def specific_energy(pos, vel, mu):
    return np.linalg.norm(vel)**2 / 2 - mu / np.linalg.norm(pos)

def eccentricity_vector(pos, vel, mu):
    v = np.append(vel,0)
    h = np.array([0,0,angular_momentum(pos, vel)])
    return (np.cross(v, h) / mu)[:2] - pos / np.linalg.norm(pos)

def eccentricity(pos, vel, mu):
    return np.linalg.norm(eccentricity_vector(pos, vel, mu))

def semi_major_axis(pos, vel, mu):
    return -mu / (2 * specific_energy(pos, vel, mu))

def argument_of_periapsis(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return np.arctan2(pos[1], pos[0])
    return np.arctan2(eccentricity_vector(pos, vel, mu)[1], eccentricity_vector(pos, vel, mu)[0])


# Anomaly calculations
def true_anomaly(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return 0
    nu = np.arccos(np.dot(eccentricity_vector(pos, vel, mu), pos) / (eccentricity(pos, vel, mu) * np.linalg.norm(pos)))
    if np.dot(pos, vel) < 0:
        nu = 2*np.pi - nu
    return nu

def eccentric_anomaly(pos, vel, mu, tol=1e-5):
    e = eccentricity(pos, vel, mu)
    a = semi_major_axis(pos, vel, mu)

    if e < tol:
        return 0
    center = -(eccentricity_vector(pos, vel, mu) / e) * e * semi_major_axis(pos, vel, mu)

    r = pos - center
    cosE = np.dot(r, center) / (np.linalg.norm(r) * np.linalg.norm(center))
    sinE = np.cross(r, center) / (np.linalg.norm(r) * np.linalg.norm(center))
    E = np.arctan2(sinE, cosE)
    if E < 0:
        E += 2*np.pi
    return E


# Apoapsis and periapsis calculations
def apoapsis(pos, vel, mu):
    return semi_major_axis(pos, vel, mu) * (1 + eccentricity(pos, vel, mu))

def periapsis(pos, vel, mu):
    return semi_major_axis(pos, vel, mu) * (1 - eccentricity(pos, vel, mu))

def apoapsis_pos(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return -pos
    return -apoapsis(pos, vel, mu) * eccentricity_vector(pos, vel, mu)/np.linalg.norm(eccentricity_vector(pos, vel, mu))

def periapsis_pos(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return pos
    return periapsis(pos, vel, mu) * eccentricity_vector(pos, vel, mu)/np.linalg.norm(eccentricity_vector(pos, vel, mu))

def apsis(pos, vel, mu):
    return (apoapsis_pos(pos, vel, mu), periapsis_pos(pos, vel, mu))


# Orbit generation functions
def random_orbit(tbr, eps=0.01, max_a=4, min_a=2, max_c=2):
    while True:
        c = np.random.uniform(tbr.r1*eps, tbr.r1*max_c)
        a = np.random.uniform(tbr.r1*min_a, tbr.r1*max_a)
        if a-c > tbr.r1*2 and a-c < tbr.r1*3:
            break
    e = random_e(c/a)
    return (e, a)

def random_e(e):
    theta = np.random.rand()*2*np.pi
    return np.array([e*np.cos(theta), e*np.sin(theta)])

def orbit_trajectory(e, a):
    c = a*np.linalg.norm(e)
    b = np.sqrt(a**2 - c**2)
    
    unit_e = e / np.linalg.norm(e)
    
    u, v = -c * unit_e
    t_rot = np.arctan2(unit_e[1], unit_e[0])

    t = np.linspace(0, 2*np.pi, 100)
    ell = np.array([a*np.cos(t), b*np.sin(t)])
    r_rot = np.array([[np.cos(t_rot) , -np.sin(t_rot)],[np.sin(t_rot) , np.cos(t_rot)]])
    
    ell = np.dot(r_rot,ell)
    ell[0,:] += u
    ell[1,:] += v
    
    return ell

def a_constrained_orbit(tbr, r, a, theta=None):
    if theta is None:
        theta = np.random.rand()*2*np.pi
    e_mag = 1 - r/a
    e = np.array([e_mag*np.cos(theta), e_mag*np.sin(theta)])
    a = a*tbr.r1
    return (e, a)


# Time calculations
def orbit_period(pos, vel, mu):
    a = semi_major_axis(pos, vel, mu)
    return 2*np.pi*np.sqrt(a**3 / mu)

def time_at_E(a, e, E, mu):
    return np.sqrt(a**3 / mu) * (E - e * np.sin(E))

def time_at_state(pos, vel, mu):
    a = semi_major_axis(pos, vel, mu)
    e = eccentricity(pos, vel, mu)
    E = eccentric_anomaly(pos, vel, mu)
    return time_at_E(a, e, E, mu)

def time_from_periapsis(pos, vel, mu):
    return time_at_state(pos, vel, mu)

def time_to_periapsis(pos, vel, mu):
    return orbit_period(pos, vel, mu) - time_from_periapsis(pos, vel, mu)

def time_to_apoapsis(pos, vel, mu):
    a = semi_major_axis(pos, vel, mu)
    e = eccentricity(pos, vel, mu)
    ta = time_at_E(a, e, np.pi, mu)
    t = time_at_state(pos, vel, mu)

    if t > ta:
        return orbit_period(pos, vel, mu) - t + ta
    return ta - t
