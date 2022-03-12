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

    if self.train or not os.path.isfile(
        "/home/aileen/heiBOX/2021_22 WS/FML/final_project/bomberman_rl/agent_code/coli_agent/q_table-2022-03-12T22:46:03.npy"
    ):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = 384  # TODO: make this dynamic
        self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 0.5
        self.exploration_rate_end = 0.05  # at end of all episodes
        self.exploration_decay_rate = 0.01  # 0.1 will reach min after ~ 100 episodes
        # Finally this will call setup_training in train.py

    else:
        self.logger.info("Loading learnt Q-Table")
        self.q_table = np.load(
            "/home/aileen/heiBOX/2021_22 WS/FML/final_project/bomberman_rl/agent_code/coli_agent/q_table-2022-03-12T22:46:03.npy"
        )


def act(self, game_state: dict) -> str:
    """Takes in the current game state and returns the chosen action in form of a string."""
    state = state_to_features(self, game_state, self.history)

    if self.train and np.random.random() < self.exploration_rate:
        self.logger.debug("Exploring")
        action = np.random.choice(ACTIONS)
        self.logger.debug(f"Action chosen: {action}")
        return action

    self.logger.debug("Exploiting")
    # TODO: Do we want to go 100% exploitation once we have learnt the q-table?
    # Alternative is to sample from the learnt q_table distribution.
    # print(state)
    self.logger.debug(f"Size q-table: {self.q_table.shape}")
    self.logger.debug(f"State: {state}")
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


def get_neighboring_tiles_until_wall(own_coord, n, game_state) -> List[Tuple[int]]:
    directions = ["N", "E", "S", "W"]
    own_coord_x, own_coord_y = own_coord[0], own_coord[1]
    all_good_fields = []

    for d, _ in enumerate(directions):
        good_fields = []
        for i in range(1, n + 1):
            try:
                if directions[d] == "N":
                    if (
                        game_state["field"][own_coord_x][own_coord_y + i] == 0
                        or game_state["field"][own_coord_x][own_coord_y + i] == 1
                    ):
                        good_fields += [(own_coord_x, own_coord_y + i)]
                    else:
                        break
                elif directions[d] == "E":
                    if (
                        game_state["field"][own_coord_x + i][own_coord_y] == 0
                        or game_state["field"][own_coord_x + i][own_coord_y] == 1
                    ):
                        good_fields += [(own_coord_x + i, own_coord_y)]
                    else:
                        break
                elif directions[d] == "S":
                    if (
                        game_state["field"][own_coord_x][own_coord_y - i] == 0
                        or game_state["field"][own_coord_x][own_coord_y - i] == 1
                    ):
                        good_fields += [(own_coord_x, own_coord_y - i)]
                    else:
                        break
                elif directions[d] == "W":
                    if (
                        game_state["field"][own_coord_x - i][own_coord_y] == 0
                        or game_state["field"][own_coord_x - i][own_coord_y] == 1
                    ):
                        good_fields += [(own_coord_x - i, own_coord_y)]
                    else:
                        break
            except IndexError:
                # print("Border")
                break

        all_good_fields += good_fields

    return all_good_fields


def state_to_features(self, game_state, history) -> np.array:
    # TODO: vectorize?
    # TODO: combine different for loops (!)
    """Parses game state to features"""
    features = np.zeros(8, dtype=np.int8)

    try:
        own_position = game_state["self"][-1]
        enemy_positions = [enemy[-1] for enemy in game_state["others"]]
    except TypeError:
        print("First game state is none")
        return

    # Feature 1: if on hot field or not
    all_hot_fields, if_dangerous = [], []
    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_pos = bomb[0]  # coordinates of bomb as type tuple
            neighbours_until_wall = get_neighboring_tiles_until_wall(
                bomb_pos, 3, game_state=game_state
            )
            if neighbours_until_wall:
                all_hot_fields += neighbours_until_wall

        if len(all_hot_fields) > 0:
            for lava in all_hot_fields:
                in_danger = own_position == lava
                if_dangerous.append(in_danger)

            features[0] = int(any(if_dangerous))
    else:
        features[0] = 0

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
            features[1 + i] = 1
        else:
            features[1 + i] = 0

    # Feature 6 ("Going to new tiles")
    num_visited_tiles = len(
        history[1]
    )  # history[2] contains agent coords of last 5 turns
    if num_visited_tiles > 1:  # otherwise the feature is and is supposed to be 0 anyway
        num_unique_visited_tiles = len(set(history[1]))
        # of 5 tiles, 3 should be new -> 60%. for start of the episode: 1 out of 2, 2 out of 3, 2 out of 4
        features[5] = (
            1 if np.floor((num_unique_visited_tiles / num_visited_tiles)) > 0.6 else 0
        )

    # Feature 7/9: amount of possibly destroyed crates: small: 0, medium: 1<4, high: >= 4
    neighbours = get_neighboring_tiles_until_wall(
        own_position, 3, game_state=game_state
    )
    crate_coordinates = []

    if neighbours:
        for coord in neighbours:
            if game_state["field"][coord[0]][coord[1]] == 1:
                crate_coordinates += [coord]

        if len(crate_coordinates) == 0:
            features[6] = 0
        elif 1 <= len(crate_coordinates) < 4:
            features[6] = 1
        elif len(crate_coordinates) >= 4:
            features[6] = 2

    else:
        features[6] = 0

    # Feature 8/10: if in opponents area
    all_enemy_fields = []
    for enemy in game_state["others"]:
        neighbours_until_wall = get_neighboring_tiles_until_wall(
            enemy[-1], 3, game_state=game_state
        )
        if neighbours_until_wall:
            all_enemy_fields += neighbours_until_wall

    if len(all_enemy_fields) > 0:
        for bad_field in all_enemy_fields:
            in_danger = own_position == bad_field
            if_dangerous.append(in_danger)

        features[7] = int(any(if_dangerous))
    else:
        features[7] = 0

    self.logger.debug(f"Feature vector: {features}")

    return features_to_state(self, features)


def features_to_state(self, feature_vector: np.array) -> int:
    # TODO: handle case that file can't be opened, read or that feature vector can't be found (currently: returns None)
    with open("indexed_state_list.csv", encoding="utf-8", mode="r") as f:
        for i, state in enumerate(f.readlines()):
            self.logger.debug(f"State lookup. Stripped state string: {state.strip()}.")
            self.logger.debug(f"Feature vector string: {str(feature_vector)}")
            if state.strip() == str(feature_vector):
                return i


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
