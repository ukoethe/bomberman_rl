import os
from collections import deque
from datetime import datetime
from typing import List, Tuple

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    # Where to put?
    self.history = [0, deque(maxlen=5)]  # [num_of_coins_collected, tiles_visited]

    if self.train or not os.path.isfile("q_table.npy"):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = 5  # TODO: make this dynamic
        self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 0.5
        self.exploration_rate_end = 0.05  # at end of all episodes
        self.exploration_decay_rate = 0.01  # 0.1 will reach min after ~ 100 episodes
        # Finally this will call setup_training in train.py

    else:
        self.logger.info("Loading learnt Q-Table")
        self.q_table = np.load("q_table.npy")


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    state = state_to_features(game_state, self.history)

    if self.train and np.random.random() < self.exploration_rate:
        self.logger.debug("Exploring")
        action = np.random.choice(ACTIONS)
        self.logger.debug(f"Action chosen: {action}")
        return action

    self.logger.debug("Exploiting")
    # TODO: Do we want to go 100% exploitation once we have learnt the q-table?
    # Alternative is to sample from the learnt q_table distribution.
    # print(state)
    action = ACTIONS[np.argmax(self.q_table[state])]
    self.logger.debug(f"Action chosen: {action}")
    return action


def _get_neighboring_tiles(own_coord, n) -> List[Tuple[int]]:
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
    features = np.zeros(5)

    try:
        own_position = game_state["self"][-1]
        enemy_positions = [enemy[-1] for enemy in game_state["others"]]
    except TypeError:
        print("First game state is none")
        return

    # Feature 2-5 ("Blockages")
    for i, neighboring_coord in enumerate(_get_neighboring_tiles(own_position, 1)):
        neighboring_x, neighboring_y = neighboring_coord
        neighboring_content = game_state["field"][neighboring_x][
            neighboring_y
        ]  # content of tile, e.g. crate=1
        explosion = (
            True
            if game_state["explosion_map"][neighboring_x][neighboring_y] != 0
            else False
        )
        ripe_bomb = False  # "ripe" = about to explode
        if (neighboring_coord, 0) in game_state["bombs"] or (
            neighboring_coord,
            1,
        ) in game_state["bombs"]:
            ripe_bomb = True
        if (
            neighboring_content != 0
            or neighboring_coord in enemy_positions
            or explosion
            or ripe_bomb
        ):
            features[i] = 1
        else:
            features[i] = 0

    # Feature 6 ("Going to new tiles")
    num_visited_tiles = len(
        history[2]
    )  # history[2] contains agent coords of last 5 turns
    if num_visited_tiles > 1:  # otherwise the feature is and is supposed to be 0 anyway
        num_unique_visited_tiles = len(set(history[2]))
        # of 5 tiles, 3 should be new -> 60%. for start of the episode: 1 out of 2, 2 out of 3, 2 out of 4
        features[5] = (
            1 if np.floor((num_unique_visited_tiles / num_visited_tiles)) > 0.6 else 0
        )

    return features_to_state(features)


def features_to_state(feature_vector: np.array) -> int:
    with open("indexed_state_list.csv", encoding="utf-8", mode="r") as f:
        for i, state in enumerate(f.readlines()):
            if state == str(feature_vector):
                return i
    return None  # TODO shouldn't happen, handle this better


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
