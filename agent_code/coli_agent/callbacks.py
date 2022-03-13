import os
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from settings import COLS, ROWS

graph = nx.Graph
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

    print(_get_graph(self, game_state))

    active_explosions = [
        index
        for index, field in np.ndenumerate(game_state["explosion_map"])
        if field != 0
    ]
    self.logger.info(f"Active explosion: {active_explosions}")
    self.logger.info(f"Bombs: {game_state['bombs']}")

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

    Considers walls, crates, active explosions and (maybe other players) as "walls", i.e. not connected"""

    # walls and crates are obstacles
    obstacles = [
        index for index, field in np.ndenumerate(game_state["field"]) if field != 0
    ]

    # TODO: Find out what works better - considering other players as obstacles (technically true) or not
    # for other_player in game_state["others"]:
    # obstacles.append(other_player[3])  # third element stores the coordinates

    active_explosions = [
        index
        for index, field in np.ndenumerate(game_state["explosion_map"])
        if field != 0
    ]
    # print(f"Active explosion: {active_explosions}")
    # print(f"Bombs: {game_state['bombs']}")
    obstacles += active_explosions

    self.logger.info(f"Obstacles: {obstacles}")

    graph = deepcopy(self.lattice_graph)

    # inplace operation
    graph.remove_nodes_from(obstacles)  # removes nodes and all edges of that node
    return graph


def _find_shortest_path(graph, a, b) -> Tuple[graph, int]:
    """Calclulates length of shortest path at current time step (without looking ahead to the future)
    between points a and b."""

    # use Djikstra to find shortest path
    shortest_path = nx.shortest_path(
        graph, source=a, target=b, weight=None, method="dijkstra"
    )
    shortest_path_length = len(shortest_path)
    return shortest_path, shortest_path_length


def _get_action(self_coord, shortest_path) -> action:
    print(self_coord)
    print(shortest_path[0])
    print(shortest_path[1])
    goal_coord = shortest_path[0]  # check if shortest_path[0] or shortest_path[1]

    if self_coord[0] == goal_coord[0]:
        if self_coord[1] + 1 == goal_coord[1]:
            return "DOWN"

        elif self_coord[1] - 1 == goal_coord[1]:
            return "UP"

    elif self_coord[1] == goal_coord[1]:
        if self_coord[0] + 1 == goal_coord[0]:
            return "RIGHT"

        elif self_coord[0] - 1 == goal_coord[1]:
            return "LEFT"


def _shortest_path_feature(self, game_state) -> action:
    graph = _get_graph(self, game_state)

    self.logger.info(f"Current Graph: {graph}")

    self_coord = game_state["self"][3]

    self.logger.info(f"Current self coord: {self_coord}")

    # no coins on board and no crates (maybe also no opponents ==> suicide?) ==> just return something
    if not any(game_state["coins"]) and not any(
        [index for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    ):
        return "UP"

    elif not any(game_state["coins"]):
        return "UP"

    # no coins available ==> calculate direction of path to nearest crate
    # elif not any(game_state["coins"]):
    #     best = (None, np.inf)

    #     crates_coordinates = [
    #         index for index, field in np.ndenumerate(game_state["field"]) if field == 1
    #     ]
    #     for crate_coord in crates_coordinates:
    #         current_path, current_path_length = _find_shortest_path(
    #             graph, self_coord, crate_coord
    #         )

    #         # not gonna get better than 1, might save a bit of computation time
    #         if current_path_length == 1:
    #             return _get_action(self_coord, current_path)

    #         elif current_path_length < best[1]:
    #             best = (current_path, current_path_length)

    #     return _get_action(self_coord, best[0])

    # calculate distance to nearest coin that no one else is closer to
    else:
        self.logger.info("There is a coin")
        coins_coordinates = game_state["coins"]
        closest_paths_to_coins = []

        # find shortest paths to all coins by all agents
        for coin_coord in coins_coordinates:
            self.logger.info(f"Looking at coin at: {coin_coord}")
            current_path, current_path_length = _find_shortest_path(
                graph, self_coord, coin_coord
            )

            for other_agent in game_state["others"]:
                best_other_agent = (None, np.inf)
                other_agent_coord = other_agent[3]
                (
                    current_path_other_agent,
                    current_path_length_other_agent,
                ) = _find_shortest_path(graph, other_agent_coord, coin_coord)

                if current_path_length_other_agent < best_other_agent[1]:
                    best_other_agent = (
                        current_path_other_agent,
                        current_path_length_other_agent,
                    )

            closest_paths_to_coins.append(
                ((current_path, current_path_length), best_other_agent)
            )

        # sort ascending by shortest length of our
        for closest_path_to_coin in closest_paths_to_coins.sort(key=lambda x: x[0][1]):
            if closest_path_to_coin[0][1] <= closest_path_to_coin[1][1]:
                return _get_action(self_coord, closest_path_to_coin[0][0])

        # if we are not closest to any coin return action that leads us to the coin we are nearest too anyway
        return _get_action(closest_paths_to_coins[0][0][0])


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

    # Position
    # maybe take out? not sure if necessary
    # features[2] = own_position[0]
    # features[3] = own_position[1]

    # Agent-Coin ratio
    # num_of_agents_left = len(game_state["others"])
    # we need to access past
    # features[4] = num_of_agents_left/num_of_coins_left

    print(_shortest_path_feature(self, game_state))

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
