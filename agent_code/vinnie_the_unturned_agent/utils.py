import numpy as np
from random import shuffle
from typing import Dict, List, Tuple
from settings import BOMB_POWER

# So we do not have to maintain this in multiple locations
ACTIONS = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    vision = list(vision_field(game_state))
    target = closest_target(game_state)

    vision.extend(list(game_state["self"][3]))
    vision.extend(list(target[0]))
    vision.append(target[1])

    # active bomb
    vision.append(game_state["self"][2])
    return tuple(vision)


def closest_target(game_state):
    _, _, _, start = game_state["self"]

    free_space = game_state["field"] == 0

    if not game_state["coins"]:
        if game_state["others"]:
            others = np.array(game_state["others"], dtype=object)
            targets = others[:, 3]
            targeting_mode = 1  # 'hostile'
        else:
            return (1, 1), 1  # if we do not have any coins or enemy's
    else:
        targets = game_state["coins"]
        targeting_mode = 0  # 'friendly'

    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None
    if isinstance(targets, np.ndarray):
        targets = targets.tolist()
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current, targeting_mode
        current = parent_dict[current]


def vision_field(game_state: Dict) -> List[Tuple]:
    # Position of the agent
    explosion_map = game_state["explosion_map"] == 1
    game_state["field"][explosion_map] = -1
    field = np.copy(game_state["field"])
    field = danger(field, game_state["bombs"])

    self_pos = game_state["self"][3]  # game_state["self"][3]

    # How far can you look
    vision = 2

    # Game Field at the position of the agent
    left = max(0, self_pos[0] - vision)
    right = min(16, self_pos[0] + vision)

    down = max(0, self_pos[1] - vision)
    top = min(16, self_pos[1] + vision)

    return field[
        left : right + 1, down : top + 1
    ].flatten()  # ToDo gleiche state größe erzwingen


def danger(field, bombs):
    # implement danger posed by bombs that are about to set off
    if bombs:
        bombs = np.array(bombs, dtype=object)[:, 0]

        # Todo optimize only one np.put
        # Todo custom event for 2

        # t = np.full((17, 17), 0)

        for pos in bombs:
            upWall = False
            downWall = False
            leftWall = False
            rightWall = False
            # dangerPosX = []
            # dangerPosY = []

            for i in range(BOMB_POWER):
                if downWall is False and field[(pos[0] - i,), pos[1]] != -1:
                    field[(pos[0] - i,), pos[1]] = 2
                    # dangerPosX.append([(pos[0] - i,)])
                else:
                    downWall = True

                if upWall is False and field[(pos[0] + i,), pos[1]] != -1:
                    field[(pos[0] + i,), pos[1]] = 2
                    # dangerPosX.append([(pos[0] + i,)])
                else:
                    upWall = True

                if leftWall is False and field[pos[0], (pos[1] - i,)] != -1:
                    field[pos[0], (pos[1] - i,)] = 2
                    # dangerPosY.append([(pos[1] - i,)])
                else:
                    leftWall = True

                if rightWall is False and field[pos[0], (pos[1] + i,)] != -1:
                    field[pos[0], (pos[1] + i,)] = 2
                    # dangerPosY.append([(pos[1] + i,)])
                else:
                    rightWall = True

            # np.put(field[pos[0], :], dangerPosY, np.full(len(dangerPosY), 2, dtype=int), mode='clip')
            # np.put(field[:, pos[1]], dangerPosX, np.full(len(dangerPosX), 2, dtype=int), mode='clip')

        return field
    return field
