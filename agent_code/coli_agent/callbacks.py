import os
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import networkx as nx
import numpy as np

from settings import COLS, ROWS

graph = nx.Graph
action = str

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup(self):
    """Sets up everything. (First call)"""

    # Where to put?
    self.history = [0, None]  # [num_of_coins_collected, tiles_visited]]

    if self.continue_training:
        print("Wow")

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

    active_explosions = [
        index
        for index, field in np.ndenumerate(game_state["explosion_map"])
        if field != 0
    ]
    # self.logger.info(f"Active explosions: {active_explosions}")
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


def _get_graph(self, game_state, crates_as_obstacles=True) -> graph:
    """Calculates the adjacency matrix of the current game state.
    Every coordinate is a node.]

    Vertex between nodes <==> both nodes are empty

    Considers walls, crates, active explosions and (maybe other players) as "walls", i.e. not connected"""

    if crates_as_obstacles:
        # walls and crates are obstacles
        obstacles = [
            index for index, field in np.ndenumerate(game_state["field"]) if field != 0
        ]

    else:
        # only walls are obstacles
        obstacles = [
            index for index, field in np.ndenumerate(game_state["field"]) if field == -1
        ]

    # TODO: Find out what works better - considering other players as obstacles (technically true) or not
    # for other_player in game_state["others"]:
    # obstacles.append(other_player[3])  # third element stores the coordinates

    active_explosions = [
        index
        for index, field in np.ndenumerate(game_state["explosion_map"])
        if field != 0
    ]

    self.logger.debug(f"Active explosions: {active_explosions}")
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
    shortest_path = None
    # use Djikstra to find shortest path
    try:
        shortest_path = nx.shortest_path(
            graph, source=a, target=b, weight=None, method="dijkstra"
        )
    except nx.exception.NodeNotFound:
        print(graph.nodes)
        raise Exception

    shortest_path_length = (
        len(shortest_path) - 1
    )  # because path considers self as part of the path
    return shortest_path, shortest_path_length


def _get_action(self_coord, shortest_path) -> action:
    goal_coord = shortest_path[1]  # 0th element is self_coord

    # x-coord is the same
    if self_coord[0] == goal_coord[0]:
        if self_coord[1] + 1 == goal_coord[1]:
            return "DOWN"

        elif self_coord[1] - 1 == goal_coord[1]:
            return "UP"

    # y-coord is the same
    elif self_coord[1] == goal_coord[1]:
        if self_coord[0] + 1 == goal_coord[0]:
            return "RIGHT"

        elif self_coord[0] - 1 == goal_coord[0]:
            return "LEFT"


def _shortest_path_feature(self, game_state) -> action:
    """
    Computes the direction along the shortest path as follows:

    If no coins and no crates exist --> random

    If no coins but a crate exists --> towards nearest crate

    If coins:

        if no coin path possible:
            towards nearest coin (thus towards first crate that's in the way)

        elif exactly one coin path possible:
            towards nearest coin # even though there might be a coin that's much closer but blocked or someone else is closer

        elif more than one coin path possible:
            try:
                towards nearest coin that no one else is more near to

            except there is no coin that our agent is nearest to:
                towards nearest coin
    """
    graph = _get_graph(self, game_state)
    self.logger.info(f"Current Graph nodes: {graph.nodes}")

    self_coord = game_state["self"][3]

    self.logger.info(f"Current self coord: {self_coord}")

    safe_coins = [
        coin
        for coin in game_state["coins"]
        if coin
        not in [
            index
            for index, field in np.ndenumerate(game_state["explosion_map"])
            if field != 0
        ]
    ]
    self.logger.info(f"Current safe coins: {safe_coins}")

    # no coins on board and no crates (maybe also no opponents ==> suicide?) ==> just return something
    if not any(safe_coins) and not any(
        [index for index, field in np.ndenumerate(game_state["field"]) if field == 1]
    ):
        return np.random.choice(ACTIONS)

    elif not any(safe_coins):
        graph = _get_graph(self, game_state, crates_as_obstacles=False)
        best = (None, np.inf)

        crates_coordinates = [
            index for index, field in np.ndenumerate(game_state["field"]) if field == 1
        ]
        for crate_coord in crates_coordinates:
            current_path, current_path_length = _find_shortest_path(
                graph, self_coord, crate_coord
            )

            # self.logger.debug(f"Current path: {current_path} with path length: {current_path_length} to crate at {crate_coord}")

            # not gonna get better than 1, might save a bit of computation time
            if current_path_length == 1:
                self.logger.debug(f"Standing directly next to crate!")
                return _get_action(self_coord, current_path)

            elif current_path_length < best[1]:
                best = (current_path, current_path_length)

        self.logger.debug(f"BEST: {best}")
        return _get_action(self_coord, best[0])

    # there is a coin
    else:
        self.logger.info("There is a safe coin and it is not *in* an explosion")
        shortest_paths_to_coins = []

        # find shortest paths to all coins by all agents
        for coin_coord in safe_coins:
            self.logger.info(f"Looking at coin at: {coin_coord}")

            try:
                current_path, current_path_length = _find_shortest_path(
                    graph, self_coord, coin_coord
                )
                current_reachable = True

            # coin path not existent
            except nx.exception.NetworkXNoPath:
                graph = _get_graph(self, game_state, crates_as_obstacles=False)
                current_path, current_path_length = _find_shortest_path(
                    graph, self_coord, coin_coord
                )
                current_reachable = False

            for other_agent in game_state["others"]:
                best_other_agent = (None, np.inf)
                other_agent_coord = other_agent[3]
                try:
                    (
                        current_path_other_agent,
                        current_path_length_other_agent,
                    ) = _find_shortest_path(graph, other_agent_coord, coin_coord)
                    current_other_agent_reachable = True

                except nx.exception.NetworkXNoPath:
                    graph = _get_graph(self, game_state, crates_as_obstacles=False)
                    (
                        current_path_other_agent,
                        current_path_length_other_agent,
                    ) = _find_shortest_path(graph, other_agent_coord, coin_coord)
                    current_other_agent_reachable = False

                # penalize with heuristic of 7 more fields if unreachable
                if not current_other_agent_reachable:
                    current_path_length_other_agent += 7

                if current_path_length_other_agent < best_other_agent[1]:
                    best_other_agent = (
                        current_path_other_agent,
                        current_path_length_other_agent,
                        current_other_agent_reachable,
                    )

            shortest_paths_to_coins.append(
                (
                    (current_path, current_path_length, current_reachable),
                    best_other_agent,
                )
            )

        # sort our [0] paths ascending by length [1]
        shortest_paths_to_coins.sort(key=lambda x: x[0][1])

        shortest_paths_to_coins_reachable = [
            shortest_path_to_coin[0][2]
            for shortest_path_to_coin in shortest_paths_to_coins
        ]

        # if none of our [0] shortest paths are actually reachable [2] we just go towards the nearest one (i.e. to its nearest crate)
        if not any(shortest_paths_to_coins_reachable):
            self.logger.debug("No coin reachable ==> Going towards nearest one")
            return _get_action(
                self_coord, shortest_paths_to_coins[0][0][0]
            )  # shortest [0] (because sorted) that is ours [0] and the actual path [0]

        # if exactly one of our [0] shortest paths is reachable [2] we go towards that one
        elif shortest_paths_to_coins_reachable.count(True) == 1:
            self.logger.debug("Exactly one coin reachable ==> Going towards that one")
            index_of_reachable_path = shortest_paths_to_coins_reachable.index(True)
            return _get_action(
                self_coord, shortest_paths_to_coins[index_of_reachable_path][0][0]
            )

        # if more than one shortest path is reachable we got towards the one that we are closest and reachable to and no one else being closer
        for shortest_path_to_coin in shortest_paths_to_coins:

            # we are able to reach it and we are closer
            if (
                shortest_path_to_coin[0][2] is True
                and shortest_path_to_coin[0][1] <= shortest_path_to_coin[1][1]
            ):
                self.logger.debug(
                    "We are able to reach a coin and we are closest to it"
                )
                return _get_action(self_coord, shortest_path_to_coin[0][0])

        self.logger.debug("Fallback Action")
        # unless we are not closest to any of our reachable coins then we return action that leads us to the coin we are nearest too anyway
        return _get_action(self_coord, shortest_paths_to_coins[0][0][0])


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

    self.logger.debug(
        f"Shortest path feature says: {_shortest_path_feature(self, game_state)}"
    )

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
