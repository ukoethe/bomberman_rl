import os
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from settings import COLS, ROWS

graph = str
action = str

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    # Where to put?
    self.history = [0, None]  # [num_of_coins_collected, tiles_visited]

    if self.train or not os.path.isfile("q_table.npy"):
        self.logger.info("Setting up Q-Learning algorithm")
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.number_of_states = 4  # TODO: make this dynamic
        self.q_table = np.zeros(shape=(self.number_of_states, len(ACTIONS)))
        self.exploration_rate_initial = 0.5
        self.exploration_rate_end = 0.05  # at end of all episodes
        self.exploration_decay_rate = 0.01  # 0.1 will reach min after ~ 100 episodes

        self.lattice_graph = nx.grid_2d_graph(m=COLS, n=ROWS)
        # Finally this will call setup_training in train.py

    else:
        self.logger.info("Loading learnt Q-Table")
        self.q_table = np.load("q_table.npy")


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


def _get_graph(self, game_state) -> graph:
    """Calculates the adjacency matrix of the current game state.
    Every coordinate is a node.]

    Vertex between nodes <==> both nodes are empty

    Considers walls, crates, other players and bombs as "walls", i.e. not connected"""

    # walls and crates are obstacles
    obstacles = [
        index for index, field in np.ndenumerate(game_state["field"]) if field != 0
    ]

    print(obstacles)

    # other players are obstacles too
    for other_player in game_state["others"]:
        obstacles.append(other_player[3])  # third element stores the coordinates

    print(obstacles)

    graph = deepcopy(self.lattice_graph)

    # inplace operation
    graph.remove_nodes_from(obstacles)  # removes nodes and all edges of that node
    return graph


def _get_shortest_path_length(game_state, x, y) -> int:
    """Calclulates length of shortest path (Manhattan distance) at current time step (without looking ahead to the future)
    between points x and y *and* considers obstacles (i.e. walls, crates, other players and bombs)."""
    graph = _get_graph(game_state)

    # use Djikstra to find shortest path
    shortest_path(graph, method="D", directed=False, unweighted=True, indices=[])
    return


def _get_shortest_path_direction(game_state, x, y) -> str:
    pass


def state_to_features(self, game_state, history) -> np.array:
    # TODO: vectorize?
    # TODO: combine different for loops (!)
    """Parses game state to features"""
    features = np.zeros(2)

    try:
        own_position = game_state["self"][-1]
    except TypeError:
        print("First game state is none")
        return

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
    features[0] = wall_counter > 2

    # Within bomb explosion zone
    # TODO shoul we have feature "bomb distance" (instead or additionally)? should that be nearest or all bombs (like enemies?)?
    # TODO take countdown into consideration?
    range_3_coordinates = _get_neighboring_tiles(own_position, 3)
    for coord in range_3_coordinates:
        try:
            bomb_present = any(
                [
                    bomb
                    for bomb in game_state["bombs"]
                    if bomb[0] in range_3_coordinates and bomb[1] != 0
                ]
            )  # access to bombs might be wrong
            features[1] = bomb_present
        except IndexError:
            print("tried to access tile out of bounds (bomb)")  # TODO remove

    print(_get_graph(self, game_state))

    # Position
    # maybe take out? not sure if necessary
    # features[2] = own_position[0]
    # features[3] = own_position[1]

    # Agent-Coin ratio
    # num_of_agents_left = len(game_state["others"])
    # we need to access past
    # features[4] = num_of_agents_left/num_of_coins_left

    if np.array_equal(features, np.array([0, 0])):
        return 0

    elif np.array_equal(features, np.array([0, 1])):
        return 1

    elif np.array_equal(features, np.array([1, 0])):
        return 2

    elif np.array_equal(features, np.array([1, 1])):
        return 3

    # return features


# Only to demonstrate test
class DecisionTransformer:
    def __init__(self):
        pass

    def adding_one(self, number: int) -> int:
        return number + 1
