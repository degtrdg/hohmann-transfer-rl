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
    f = true_anomaly(pos, vel, mu, tol)

    if e < tol:
        return 0
    center = -(eccentricity_vector(pos, vel, mu) / e) * e * semi_major_axis(pos, vel, mu)

    cosE = (e + np.cos(f)) / (1 + e*np.cos(f))
    sinE = np.sqrt(1-e**2)*np.sin(f)/(1+e*np.cos(f))
    E = np.arctan2(sinE, cosE)
    if E < 0:
        E = 2*np.pi + E
    return E

def eccentric_anomaly_from_f(e, f):
    cosE = (e + np.cos(f)) / (1 + e*np.cos(f))
    sinE = np.sqrt(1-e**2)*np.sin(f)/(1+e*np.cos(f))
    E = np.arctan2(sinE, cosE)
    if E < 0:
        E = 2*np.pi + E
    return E

def target_relative_anomaly(pos, target):
    e = target[0]
    theta = np.arctan2(pos[1], pos[0]) - np.arctan2(e[1], e[0])
    if theta < 0:
        theta += 2*np.pi
    return theta

def e_angle_diff(pos, vel, mu, target):
    e = eccentricity_vector(pos, vel, mu)
    if np.linalg.norm(e) < 1e-2:
        direction = -np.sign(np.cross(pos, target[0]))
        e_angle_diff = direction * np.arccos(np.dot(pos, target[0])/(np.linalg.norm(pos)*np.linalg.norm(target[0])))
    else:
        direction = -np.sign(np.cross(e, target[0]))
        e_angle_diff = direction * np.arccos(np.dot(e, target[0])/(np.linalg.norm(e)*np.linalg.norm(target[0])))
    return e_angle_diff

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
        theta = np.random.uniform(np.pi/4, 7*np.pi/4)
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

def time_at_state(pos, vel, mu, tol=1e-5):
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
        # print("t > ta", t, ta)
        return orbit_period(pos, vel, mu) - t + ta
    else:
        # print("t < ta", t, ta)
        return ta - t
    
def time_from_apoapsis(pos, vel, mu):
    a = semi_major_axis(pos, vel, mu)
    e = eccentricity(pos, vel, mu)
    ta = time_at_E(a, e, np.pi, mu)
    t = time_at_state(pos, vel, mu)
    if t > ta:
        return t - ta
    else:
        return orbit_period(pos, vel, mu) - ta + t

def time_to_E(pos, vel, mu, E):
    a = semi_major_axis(pos, vel, mu)
    e = eccentricity(pos, vel, mu)
    te = time_at_E(a, e, E, mu)
    t = time_at_state(pos, vel, mu)
    if t > te:
        return orbit_period(pos, vel, mu) - t + te
    else:
        return te - t

def time_from_E(pos, vel, mu, E):
    a = semi_major_axis(pos, vel, mu)
    e = eccentricity(pos, vel, mu)
    te = time_at_E(a, e, E, mu)
    t = time_at_state(pos, vel, mu)
    if t > te:
        return t - te
    else:
        return orbit_period(pos, vel, mu) - te + t

def time_to_target(pos, vel, mu, target, tol=1e-5):
    e_vec = eccentricity_vector(pos, vel, mu)
    e = np.linalg.norm(e_vec)
    target_e = target[0]

    if np.linalg.norm(e_vec) < tol:
        target_f = np.arctan2(target_e[1], target_e[0]) - np.arctan2(pos[1], pos[0])
    else:
        target_f = np.arctan2(target_e[1], target_e[0]) - np.arctan2(e_vec[1], e_vec[0])
    target_f = -target_f

    if target_f < 0:
        target_f += 2*np.pi

    target_E = eccentric_anomaly_from_f(e, target_f)
    tE = time_to_E(pos, vel, mu, target_E)
    return tE

def time_from_target(pos, vel, mu, target, tol=1e-5):
    e_vec = eccentricity_vector(pos, vel, mu)
    e = np.linalg.norm(e_vec)
    target_e = target[0]

    if np.linalg.norm(e_vec) < tol:
        target_f = np.arctan2(target_e[1], target_e[0]) - np.arctan2(pos[1], pos[0])
    else:
        target_f = np.arctan2(target_e[1], target_e[0]) - np.arctan2(e_vec[1], e_vec[0])
    target_f = -target_f

    if target_f < 0:
        target_f += 2*np.pi

    target_E = eccentric_anomaly_from_f(e, target_f)
    tE = time_from_E(pos, vel, mu, target_E)
    return tE
    
def absolute_target_time(pos, vel, mu, target):
    t_to = time_to_target(pos, vel, mu, target)
    t_from = time_from_target(pos, vel, mu, target)
    if t_to < t_from:
        return t_to
    else:
        return -t_from

