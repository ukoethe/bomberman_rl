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


def _get_neighboring_tiles(own_coord, n):
    own_coord_x = own_coord[0]
    own_coord_y = own_coord[1]
    neighboring_coordinates = []
    for i in range(1, n + 1):
        neighboring_coordinates.append(
            (own_coord_x, own_coord_y + i)
        )  # down in the matrix
        neighboring_coordinates.append(
            (own_coord_x, own_coord_y - i)
        )  # up in the matrix
        neighboring_coordinates.append(
            (own_coord_x + i, own_coord_y)
        )  # right in the matrix
        neighboring_coordinates.append(
            (own_coord_x - i, own_coord_y)
        )  # left in the matrix
    return neighboring_coordinates


def state_to_features(game_state, history) -> np.array:
    # TODO: vectorize?
    # TODO: combine different for loops (!)
    """Parses game state to features"""
    features = np.zeros(4)

    own_position = game_state["self"][-1]

    # How many walls
    wall_counter = 0
    neighboring_coordinates = _get_neighboring_tiles(own_position, 1)
    for coord in neighboring_coordinates:
        try:
            if game_state["field"][coord] == -1:  # geht das? wer weiß
                wall_counter += 1
        except IndexError:
            print(
                "tried to access tile out of bounds (walls)"
            )  # TODO remove, just for "debugging"
    features[0] = wall_counter

    # Within bomb explosion zone
    # TODO shoul we have feature "bomb distance" (instead or additionally)? should that be nearest or all bombs (like enemies?)?
    # TODO take countdown into consideration?
    range_3_coordinates = _get_neighboring_tiles(own_position, 3)
    for coord in range_3_coordinates:
        try:
            bomb_present = (
                True if game_state["bombs"][coord][0] != 0 else False
            )  # access to bombs might be wrong
            features[1] = bomb_present
        except IndexError:
            print("tried to access tile out of bounds (bomb)")  # TODO remove

    # Position
    # maybe take out? not sure if necessary
    features[2] = own_position[0]
    features[3] = own_position[1]

    # Agent-Coin ratio
    # num_of_agents_left = len(game_state["others"])
    # we need to access past
    # features[4] = num_of_agents_left/num_of_coins_left

    return features


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
