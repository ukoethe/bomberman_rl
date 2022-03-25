import numpy as np
from random import shuffle
from typing import Dict, List, Tuple
from settings import BOMB_POWER
from math import cos, sin


# So we do not have to maintain this in multiple locations
ACTIONS = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])
rotation_param = 0
pivotPoint_param = (1, 1)
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
    feature = list()
    vision = vision_field(game_state)  # The Game we can see
    target = closest_target(
        game_state
    )  # In which direction is the next target and in what 'mode' are we
    ownPosition = rotate_and_transform(game_state["self"][3])  # Where are we

    feature.extend(vision)
    feature.extend(list(ownPosition))
    feature.extend(list(target[0]))
    feature.append(target[1])
    feature.append(int(game_state["self"][2]))  # Can we set a bomb ?

    # return np.array([vision, list(ownPosition), list(target[0]), target[1], int(game_state["self"][2])]).flatten()
    return feature
    # assert len(feature) == 31, "this cant be"
    # return tuple(feature)


def predict_input(input):
    # Forest Classifier expects 2D array for multiple predicts at once but we only do one at a time
    return np.array(input).reshape(1, -1)


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
            return rotate_and_transform(current), targeting_mode
        current = parent_dict[current]


def target_others(
    self, myPoints: int, otherPoints: List, collectableCoins: int
) -> bool:
    # cant win with only collecting coins or do not need to collect coins anymore

    if len(otherPoints) == 3:
        self.currentCoins = 9 - (
            myPoints + sum(otherPoints)
        )  # 9 oder 50 bei Coin_heaven #ToDo target crates
    else:
        self.currentCoins = self.currentCoins

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

    elif game_state["self"][3] == (1, 15):
        # Rechtsrotation unten links
        ACTIONS = np.array(["LEFT", "UP", "RIGHT", "DOWN", "WAIT", "BOMB"])
        matrix_rot_param = 3
        rotation_param = np.deg2rad(90)
        transformation_param[0] = 0
        transformation_param[1] = -14

    elif game_state["self"][3] == (15, 1):
        # Linksrotation oben rechts
        ACTIONS = np.array(["RIGHT", "DOWN", "LEFT", "UP", "WAIT", "BOMB"])
        matrix_rot_param = 1
        rotation_param = np.deg2rad(-90)
        transformation_param[0] = -14
        transformation_param[1] = 0

    elif game_state["self"][3] == (15, 15):
        # 180 Grad unten rechts
        ACTIONS = np.array(["DOWN", "LEFT", "UP", "RIGHT", "WAIT", "BOMB"])
        rotation_param = np.deg2rad(-180)
        matrix_rot_param = 2
        transformation_param[0] = -14
        transformation_param[1] = -14

    return ACTIONS


def rotate_and_transform(xy):
    x, y = xy
    cx, cy = pivotPoint_param
    px, py = x + transformation_param[0], y + transformation_param[1]

    px -= cx
    py -= cy

    xnew = round(px * cos(rotation_param) - py * sin(rotation_param))
    ynew = round(px * sin(rotation_param) + py * cos(rotation_param))

    px = xnew + cx
    py = ynew + cy

    return px, py


def vision_field(game_state: Dict) -> List[Tuple]:
    # Position of the agent
    explosion_map = game_state["explosion_map"] == 1
    game_state["field"][explosion_map] = -1
    field = np.rot90(
        danger(np.copy(game_state["field"]), game_state["bombs"]), k=matrix_rot_param
    )  # game_state["field"]

    self_pos = rotate_and_transform(game_state["self"][3])  # game_state["self"][3]

    # How far can you look
    vision = 2

    # Game Field at the position of the agent
    left = max(0, self_pos[0] - vision)
    right = min(16, self_pos[0] + vision)

    down = max(0, self_pos[1] - vision)
    top = min(16, self_pos[1] + vision)

    vision_field = field[left : right + 1, down : top + 1]

    # Thanks to this the features always have the same shape
    if self_pos[0] - vision < 0:
        vision_field = np.insert(vision_field, 0, values=-1, axis=0)
    elif self_pos[0] + vision > 16:
        vision_field = np.insert(vision_field, -1, values=-1, axis=0)
    if self_pos[1] - vision < 0:
        vision_field = np.insert(vision_field, 0, values=-1, axis=1)
    elif self_pos[1] + vision > 16:
        vision_field = np.insert(vision_field, -1, values=-1, axis=1)

    # assert len(vision_field.flatten()) == 25, "somethings off"
    return vision_field.flatten()


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
