import pygame
import numpy as np
from SimpleBurnEnv import SimpleBurnEnv
import orbital_mechanics as om

TRAIL_COLOR = (255, 0, 0)  # Red color for the trail
STEP = 1  # Draw a trail dot every 5 frames
MAX_TRAIL_LENGTH = 500  # Maximum number of points in the trail
PREDICTED_ORBIT_COLOR = (0, 255, 0)  # Green color for the orbit
TARGET_ORBIT_COLOR = (255, 255, 255)  # White color for the orbit
# Scaling factor for eccentricity vector
ECCENTRICITY_SCALE = 200

# Scale factors for game dimensions and elements
SCALE_FACTOR = 0.5  # adjust this value to get the right fit on your screen
PADDING = 0.25  # 10% padding

# Game dimensions
# WIDTH, HEIGHT = int(SCREEN_WIDTH * SCALE_FACTOR), int(SCREEN_HEIGHT * SCALE_FACTOR)
WIDTH, HEIGHT = int(1000), int(1000)

# Constants for the game
FPS = 60  # Frames per second
ROCKET_RADIUS = int(5 * SCALE_FACTOR)  # Size of the rocket on screen, scaled
EARTH_RADIUS = int(10 * SCALE_FACTOR)  # Size of earth on screen, scaled

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))

# Create font object
font_size = 16
font_color = (255, 255, 255)  # White
font = pygame.font.Font(None, font_size)  # Default font

# Center point for drawing
CENTER_POINT = np.array([WIDTH, HEIGHT]) / 2

# Create the environment
env = SimpleBurnEnv()
state, info = env.reset()

# Run the game loop
running = True
prev_states = []
prev_thrusts = []
iteration = 0
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Rocket control
    keys = pygame.key.get_pressed()
    action = 1 if keys[pygame.K_SPACE] else 0
    # reset the environment if the 'r' key is pressed
    if keys[pygame.K_r]:
        env.reset()
        prev_states = []
        prev_thrusts = []

    # Step the environment
    state, reward, done, truncated, info = env.step(action)
    if done:
        # Pause the game
        # running = False
        env.reset()
        prev_states = []
        prev_thrusts = []
        # break

    e = env.orbit_state[1:3]
    a = env.orbit_state[3]
    predicted_orbit = om.orbit_trajectory(e, a) # ellipse
    predicted_orbit = (predicted_orbit / (2*env.a0)).T * (CENTER_POINT * (1 - 2 * PADDING))
    predicted_orbit[:,0] = CENTER_POINT[0] + predicted_orbit[:,0]
    predicted_orbit[:,1] = CENTER_POINT[1] - predicted_orbit[:,1]
    predicted_eccentricity_vector = e * ECCENTRICITY_SCALE
    predicted_eccentricity_vector[1] = -predicted_eccentricity_vector[1]
    target_orbit = om.orbit_trajectory(env.target[0], env.target[1])
    target_orbit = (target_orbit / (2*env.a0)).T * (CENTER_POINT * (1 - 2 * PADDING))
    target_orbit[:,0] = CENTER_POINT[0] + target_orbit[:,0]
    target_orbit[:,1] = CENTER_POINT[1] - target_orbit[:,1]
    target_eccentricity_vector = env.target[0] * ECCENTRICITY_SCALE
    target_eccentricity_vector[1] = -target_eccentricity_vector[1]

    # Draw text for state variables and environment attributes
    texts = [
        f'Anomaly: {env.state[0]}',
        f'Delta eccentricity x: {env.state[1]}',
        f'Delta eccentricity y: {env.state[2]}',
        f'Delta semi-major axis: {env.state[3]}',
        f'Previous action: {env.state[4]}',
        f'Time step: {env.t0}',
        f'Reward: {reward}',
        f'Current eccentricity vector: {env.orbit_state[1:3]}',
        f'Current eccentricity vector magnitude: {np.linalg.norm(env.orbit_state[1:3])}',
        f'Target eccentricity: {env.target[0]}',
        f'Current semi-major axis length: {env.orbit_state[3]}',
        f'Target semi-major axis length: {env.target[1]}',
    ]
    # Store the previous states
    if iteration % STEP == 0 and iteration != 0:
        prev_states.append(rocket_position.copy())
        prev_thrusts.append(action)
        if len(prev_states) > MAX_TRAIL_LENGTH:
            prev_states.pop(0)  # Remove the oldest point
            prev_thrusts.pop(0)
        # Clear the screen
    if iteration % 5 == 0 and iteration != 0:
        win.fill((0, 0, 0))
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, font_color)
            win.blit(text_surface, (10, 10 + i * (font_size + 5)))  # Padding of 5 between lines
      
    # Draw Earth at the center
    pygame.draw.circle(win, BLUE, CENTER_POINT.astype(int), EARTH_RADIUS)

    # Draw the predicted orbit
    pygame.draw.lines(win, PREDICTED_ORBIT_COLOR, False, predicted_orbit.astype(int), 1)

    # Draw the target orbit
    pygame.draw.lines(win, TARGET_ORBIT_COLOR, False, target_orbit.astype(int), 1)

    # Draw the eccentricity vectors
    pygame.draw.line(win, PREDICTED_ORBIT_COLOR, CENTER_POINT.astype(int), 
                     (CENTER_POINT + predicted_eccentricity_vector).astype(int), 2)

    pygame.draw.line(win, TARGET_ORBIT_COLOR, CENTER_POINT.astype(int), 
                     (CENTER_POINT + target_eccentricity_vector).astype(int), 2)

    # Draw the rocket's path
    for point, thrust in zip(prev_states, prev_thrusts):
        if thrust == 1:
            pygame.draw.circle(win, YELLOW, point.astype(int), ROCKET_RADIUS)
        else:
            pygame.draw.circle(win, TRAIL_COLOR, point.astype(int), ROCKET_RADIUS)

    # Draw the rocket
    rocket_position = env.ivp_state[:2]/((2*env.a0)) * (CENTER_POINT * (1 - 2 * PADDING))
    rocket_position[0] = CENTER_POINT[0] + rocket_position[0]
    rocket_position[1] = CENTER_POINT[1] - rocket_position[1]
    pygame.draw.circle(win, WHITE, rocket_position.astype(int), ROCKET_RADIUS)

    # Draw the line between the origin and the rocket
    pygame.draw.line(win, TARGET_ORBIT_COLOR, CENTER_POINT.astype(int), rocket_position.astype(int), 2)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(FPS)

    # Increment the iteration
    iteration += 1

# Clean up
pygame.quit()
