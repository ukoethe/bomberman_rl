import os

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    # Where to put?
    self.history = [0, None]  # [num_of_coins_collected, tiles_visited]

    if self.train or not os.path.isfile("q_table.npy"):
        self.logger.info("Setting up Q-Learning algorithm")
        self.number_of_states = 4  # TODO: make this dynamic
        self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 0.5
        self.exploration_rate_end = 0.05  # at end of all episodes
        self.exploration_decay_rate = 0.05  # 0.1 will reach min after ~ 100 episodes
        # Finally this will call setup_training in train.py

    else:
        self.logger.info("Loading learnt Q-Table")
        self.q_table = np.load("q_table.npy")


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    state = state_to_features(game_state, self.history)
    if self.train and np.random.random() < self.exploration_rate:
        self.logger.debug("Exploring")
        return np.random.choice(ACTIONS)

    self.logger.debug("Exploiting")
    # TODO: Do we want to go 100% exploitation once we have learnt the q-table?
    # Alternative is to sample from the learnt q_table distribution.
    return ACTIONS[np.argmax(self.q_table[state])]


def state_to_features(game_state):
    pass


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
