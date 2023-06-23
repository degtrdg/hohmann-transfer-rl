# OpenAI Gym

- Is an API for single agent reinforcement learning environments
- Has 4 main functions:
  - `make` - creates an environment
  - `reset` - resets the environment
  - `step` - takes an action and returns the next state, reward, and whether the episode is done
  - `render` - renders the environment

Main class

- `Env` - the main class that all environments inherit from
  - It represents a markov decision process
- `Wrappers` - can change the result of the `Env` before it reaches the user
