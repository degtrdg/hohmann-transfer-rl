from SimpleBurnEnv import SimpleBurnEnv
import pygame

def main():
    # Create an environment
    env = SimpleBurnEnv()

    # Reset the environment
    env.reset()

    # Render the environment
    while True:
        env.render()

if __name__ == "__main__":
    env = SimpleBurnEnv()
    env.reset()
    while True:
        env.render()
        pygame.time.delay(100)