import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    # Where to put?
    self.history = [0, None]  # [num_of_coins_collected, tiles_visited]

    if self.train or not os.path.isfile("q_table.npy"):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = 12  # TODO: make this dynamic
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


def get_neighboring_tiles_except_walls(own_coord, n, game_state):
    directions = ["N", "E", "S", "W"]
    own_coord_x, own_coord_y = own_coord[0], own_coord[1]
    all_good_fields = []

    for d in range(len(directions)):
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


def state_to_features(game_state, history) -> np.array:
    # TODO: vectorize?
    # TODO: combine different for loops (!)
    """Parses game state to features"""
    features = [0] * 3

    try:
        own_position = game_state["self"][-1]
    except TypeError:
        print("First game state is none")
        return

    # # How many walls
    # wall_counter = 0
    # neighboring_coordinates = _get_neighboring_tiles(own_position, 1)
    # for coord in neighboring_coordinates:
    #     try:
    #         if game_state["field"][coord] == -1:  # geht das? wer weiß
    #             wall_counter += 1
    #     except IndexError:
    #         print(
    #             "tried to access tile out of bounds (walls)"
    #         )  # TODO remove, just for "debugging"
    # features[0] = wall_counter > 2

    # # Within bomb explosion zone
    # # TODO shoul we have feature "bomb distance" (instead or additionally)? should that be nearest or all bombs (like enemies?)?
    # # TODO take countdown into consideration?
    # range_3_coordinates = _get_neighboring_tiles(own_position, 3)
    # for coord in range_3_coordinates:
    #     try:
    #         bomb_present = any(
    #             [
    #                 bomb
    #                 for bomb in game_state["bombs"]
    #                 if bomb[0] in range_3_coordinates and bomb[1] != 0
    #             ]
    #         )  # access to bombs might be wrong
    #         features[1] = bomb_present
    #     except IndexError:
    #         print("tried to access tile out of bounds (bomb)")  # TODO remove

    # # Position
    # # maybe take out? not sure if necessary
    # # features[2] = own_position[0]
    # # features[3] = own_position[1]

    # # Agent-Coin ratio
    # # num_of_agents_left = len(game_state["others"])
    # # we need to access past
    # # features[4] = num_of_agents_left/num_of_coins_left

    # if np.array_equal(features, np.array([0, 0])):
    #     return 0

    # elif np.array_equal(features, np.array([0, 1])):
    #     return 1

    # elif np.array_equal(features, np.array([1, 0])):
    #     return 2

    # elif np.array_equal(features, np.array([1, 1])):
    #     return 3

    # Feature 1: if on hot field or not
    all_hot_fields, if_dangerous = [], []
    if len(game_state["bombs"]) > 0:
        for bomb in game_state["bombs"]:
            bomb_pos = bomb[0]  # coordinates of bomb as type tuple
            neighbours_except_walls = get_neighboring_tiles_except_walls(
                bomb_pos, 3, game_state=game_state
            )
            if neighbours_except_walls != None:
                all_hot_fields += neighbours_except_walls

        if len(all_hot_fields) > 0:
            for lava in all_hot_fields:
                in_danger = own_position == lava
                if_dangerous.append(in_danger)

            features[0] = int(any(if_dangerous))
    else:
        features[0] = 0

    # Feature 9: amount of possibly destroyed crates: small: 0, medium: 1<4, high: >= 4
    neighbours = get_neighboring_tiles_except_walls(
        own_position, 3, game_state=game_state
    )
    crate_coordinates = []

    if neighbours != None:
        for coord in neighbours:
            if game_state["field"][coord[0]][coord[1]] == 1:
                crate_coordinates += [coord]

        if len(crate_coordinates) == 0:
            features[1] = 1
        elif 1 <= len(crate_coordinates) < 4:
            features[1] = 2
        elif len(crate_coordinates) >= 4:
            features[1] = 3

    else:
        features[1] = 0

    # Feature 10: if in opponents area
    all_enemy_fields = []
    for enemy in game_state["others"]:
        neighbours_except_walls = get_neighboring_tiles_except_walls(
            enemy[-1], 3, game_state=game_state
        )
        if neighbours_except_walls != None:
            all_enemy_fields += neighbours_except_walls

    if len(all_enemy_fields) > 0:
        for bad_field in all_enemy_fields:
            in_danger = own_position == bad_field
            if_dangerous.append(in_danger)

        features[2] = int(any(if_dangerous))
    else:
        features[2] = 0

    # return features

    # return None

    # return 10000


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
