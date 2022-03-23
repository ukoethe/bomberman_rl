import numpy as np
from collections import defaultdict
from random import shuffle
from typing import Dict, List, Tuple
from settings import BOMB_POWER, SCENARIOS
from math import cos, sin, pi


# So we do not have to maintain this in multiple locations
ACTIONS = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])
rotation_param = 0
matrix_rot_param = 0
transformation_param = np.array([0, 0])


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

    vision.extend(list(target))

    # active bomb
    vision.append(game_state["self"][2])

    return tuple(vision)


def closest_target(game_state):
    _, _, _, start = game_state["self"]

    free_space = game_state["field"] == 0

    others = np.array(game_state["others"], dtype=object)
    targeting_mode = 0  # 'friendly'

    # What should be prioritised
    if (
        isinstance(others, np.ndarray)
        and others.any()
        and target_others(game_state["self"][1], others[:, 1], len(game_state["coins"]))
        or not game_state["coins"]
    ):
        if not isinstance(others[:, 3], np.ndarray):
            return (0, 0), 1  # if we do not have any others
        else:
            targets = others[:, 3]
            targeting_mode = 1  # 'hostile'
    else:
        targets = game_state["coins"]

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
            return rotate_and_transform(current), targeting_mode
        current = parent_dict[current]


def target_others(myPoints: int, otherPoints: [], collectableCoins: int) -> bool:
    # cant win with only collecting coins or do not need to collect coins anymore
    if not isinstance(otherPoints, np.ndarray):
        return False

    return (
        myPoints + collectableCoins < max(otherPoints)
        or collectableCoins + max(otherPoints) < myPoints
    )


def relative_position_coins(items: List[Tuple]) -> List[Tuple]:
    """
    Takes the original coordinates and recalculates them relative to the start position defined in action_rotation
    This limits possible states without losing information
    """
    # TODO: Clean this up, the slight alteration item[0] is needed because bomb tuple is ((X,Y),Turns) and Coins just (X,Y)
    # TODO: Only take in vision / target
    return [rotate_and_transform(item) for item in items]


def relative_position_bombs(items: List[Tuple]) -> List[Tuple]:
    return [(rotate_and_transform(item[0]), item[1]) for item in items]


def action_rotation(game_state: Dict):
    # TODO: Relative to agent
    """
    #IDEA not used for now

    Rotates all actions as if the agent always starts on the top left.
    This lets every start always look similar and should result in more stable starting strategy
    """
    global ACTIONS
    global rotation_param
    global matrix_rot_param
    global transformation_param

    if game_state["self"][3] == (1, 1):
        # Ausgangspunkt oben links
        ACTIONS = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])
        rotation_param = 0
        matrix_rot_param = 0
        transformation_param[0] = 0
        transformation_param[1] = 0

    if game_state["self"][3] == (15, 1):
        # Rechtsrotation unten links
        ACTIONS = np.array(["LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"])
        rotation_param = pi / 180 * -90
        matrix_rot_param = 3
        transformation_param[0] = -15
        transformation_param[1] = 0

    if game_state["self"][3] == (1, 15):
        # Linksrotation oben rechts
        ACTIONS = np.array(["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"])
        rotation_param = pi / 180 * 90
        matrix_rot_param = 1
        transformation_param[0] = 0
        transformation_param[1] = -15

    if game_state["self"][3] == (15, 15):
        # 180 Grad unten rechts
        ACTIONS = np.array(["DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"])
        rotation_param = pi / 180 * 180
        matrix_rot_param = 2
        transformation_param[0] = -15
        transformation_param[1] = -15


def rotate_and_transform(xy):
    x, y = xy
    x, y = x - transformation_param[0], y - transformation_param[1]
    x = int(x * cos(rotation_param) - y * sin(rotation_param))
    y = int(x * sin(rotation_param) + y * cos(rotation_param))

    return x, y


def vision_field(game_state: Dict) -> List[Tuple]:
    # Position of the agent
    explosion_map = game_state["explosion_map"] == 1
    game_state["field"][explosion_map] = -1
    field = np.rot90(danger(game_state), k=matrix_rot_param)  # game_state["field"]
    self_pos = rotate_and_transform(game_state["self"][3])  # game_state["self"][3]

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


def danger(game_state: Dict):
    # implement danger posed by bombs that are about to set off
    if game_state["bombs"]:
        bombs = np.array(game_state["bombs"])[:, 0]
        field = game_state["field"]
        dangerPosX = []
        dangerPosY = []

        # Todo optimize only one np.put
        # Todo custom event for 2

        for pos in bombs:
            upWall = False
            downWall = False
            leftWall = False
            rightWall = False

            for i in range(BOMB_POWER):
                if downWall is False and field[(pos[0] - i,), pos[1]] == 0:
                    dangerPosY.append([(pos[0] - i,)])
                else:
                    downWall = True

                if upWall is False and field[(pos[0] + i,), pos[1]] == 0:
                    dangerPosY.append([(pos[0] + i,)])
                else:
                    upWall = True

                if leftWall is False and field[pos[0], (pos[1] - i,)] == 0:
                    dangerPosX.append([(pos[1] - i,)])
                else:
                    leftWall = True

                if rightWall is False and field[pos[0], (pos[1] + i,)] == 0:
                    dangerPosX.append([(pos[1] + i,)])
                else:
                    rightWall = True

        for pos in bombs:
            np.put(
                field[pos[0], :],
                dangerPosX,
                np.full(len(dangerPosX), 2, dtype=int),
                mode="clip",
            )
            np.put(
                field[:, pos[1]],
                dangerPosY,
                np.full(len(dangerPosY), 2, dtype=int),
                mode="clip",
            )

        return field
    return game_state["field"]
