import pygame
import numpy as np
from SimpleBurnEnv import SimpleBurnEnv
import orbital_mechanics as om

TRAIL_COLOR = (255, 0, 0)  # Red color for the trail
STEP = 5  # Draw a trail dot every 5 frames
MAX_TRAIL_LENGTH = 500  # Maximum number of points in the trail
PREDICTED_ORBIT_COLOR = (0, 255, 0)  # Green color for the orbit
TARGET_ORBIT_COLOR = (0, 0, 255)  # Red color for the orbit

# Scale factors for game dimensions and elements
SCALE_FACTOR = 0.5  # adjust this value to get the right fit on your screen
PADDING = 0.25  # 10% padding

# Game dimensions
# WIDTH, HEIGHT = int(SCREEN_WIDTH * SCALE_FACTOR), int(SCREEN_HEIGHT * SCALE_FACTOR)
WIDTH, HEIGHT = int(1000), int(800)

# Constants for the game
FPS = 60  # Frames per second
ROCKET_RADIUS = int(5 * SCALE_FACTOR)  # Size of the rocket on screen, scaled
EARTH_RADIUS = int(50 * SCALE_FACTOR)  # Size of earth on screen, scaled

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))

# Center point for drawing
CENTER_POINT = np.array([WIDTH, HEIGHT]) / 2

# Create the environment
env = SimpleBurnEnv()
state, info = env.reset()

# Run the game loop
running = True
prev_states = []
iteration = 0
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Rocket control
    keys = pygame.key.get_pressed()
    action = 1 if keys[pygame.K_SPACE] else 0

    # Step the environment
    state, reward, done, truncated, info = env.step(action)

    e = env.orbit_state[1:3]
    a = env.orbit_state[3]
    predicted_orbit = om.orbit_trajectory(e, a) # ellipse
    predicted_orbit = (predicted_orbit / env.a0).T * (CENTER_POINT * (1 - 2 * PADDING)) + CENTER_POINT
    target_orbit = om.orbit_trajectory(env.target[0],env.target[1])
    target_orbit = (target_orbit / env.a0).T * (CENTER_POINT * (1 - 2 * PADDING)) + CENTER_POINT

    # Store the previous states
    if iteration % STEP == 0 and iteration != 0:
        prev_states.append(rocket_position.copy())
        if len(prev_states) > MAX_TRAIL_LENGTH:
            prev_states.pop(0)  # Remove the oldest point
        # Clear the screen
        win.fill((0, 0, 0))
    
    # Draw Earth at the center
    pygame.draw.circle(win, BLUE, CENTER_POINT.astype(int), EARTH_RADIUS)

    # Draw the predicted orbit
    pygame.draw.lines(win, PREDICTED_ORBIT_COLOR, False, predicted_orbit.astype(int), 1)

    # Draw the target orbit
    pygame.draw.lines(win, TARGET_ORBIT_COLOR, False, target_orbit.astype(int), 1)

    # Draw the rocket's path
    for point in prev_states:
        pygame.draw.circle(win, TRAIL_COLOR, point.astype(int), ROCKET_RADIUS)

    # Draw the rocket
    rocket_position = env.ivp_state[:2]/(env.a0) * (CENTER_POINT * (1 - 2 * PADDING)) + CENTER_POINT
    pygame.draw.circle(win, WHITE, rocket_position.astype(int), ROCKET_RADIUS)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(FPS)

    # Increment the iteration
    iteration += 1

# Clean up
pygame.quit()
