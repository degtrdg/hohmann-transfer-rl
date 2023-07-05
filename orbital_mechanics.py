import numpy as np

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

def argument_of_periapsis(pos, vel, mu):
    return np.arctan2(eccentricity_vector(pos, vel, mu)[1], eccentricity_vector(pos, vel, mu)[0])

def true_anomaly(pos, vel, mu):
    return np.arctan2(np.dot(pos, eccentricity_vector(pos, vel, mu)), np.dot(pos, vel))

def apoapsis(pos, vel, mu):
    return semi_major_axis(pos, vel, mu) * (1 + eccentricity(pos, vel, mu))

def periapsis(pos, vel, mu):
    return semi_major_axis(pos, vel, mu) * (1 - eccentricity(pos, vel, mu))

def apoapsis_pos(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return np.array([apoapsis(pos, vel, mu),0])
    return -apoapsis(pos, vel, mu) * eccentricity_vector(pos, vel, mu)/np.linalg.norm(eccentricity_vector(pos, vel, mu))

def periapsis_pos(pos, vel, mu, tol=1e-5):
    if eccentricity(pos, vel, mu) < tol:
        return np.array([-periapsis(pos, vel, mu), 0])
    return periapsis(pos, vel, mu) * eccentricity_vector(pos, vel, mu)/np.linalg.norm(eccentricity_vector(pos, vel, mu))

def apsis(pos, vel, mu):
    return (apoapsis_pos(pos, vel, mu), periapsis_pos(pos, vel, mu))

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

def nu(pos):
    theta = np.arctan2(pos[1], pos[0])
    if theta < 0:
        theta += 2*np.pi
    return theta

def a_constrained_orbit(tbr, r, a, theta=None):
    if theta is None:
        theta = np.random.rand()*2*np.pi
    e_mag = 1 - r/a
    e = np.array([e_mag*np.cos(theta), e_mag*np.sin(theta)])
    a = a*tbr.r1
    return (e, a)


"""
gravitational_constant = 6.67408e-11
earth_mass = 5.972e24
earth_radius = 6371000

pos = np.array([earth_radius*1.1, 0])
vel = np.array([0, -7700])
mu = gravitational_constant * earth_mass

print("Angular momentum: ", angular_momentum(pos, vel))
print("Specific energy: ", specific_energy(pos, vel, mu))
print("Eccentricity vector: ", eccentricity_vector(pos, vel, mu))
print("Eccentricity: ", eccentricity(pos, vel, mu))
print("Apoapsis: ", apoapsis(pos, vel, mu))
print("Periapsis: ", periapsis(pos, vel, mu))
"""