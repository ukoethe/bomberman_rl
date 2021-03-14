import random
import agent_code.auto_bomber.auto_bomber_config as config

import numpy as np

from agent_code.auto_bomber.model import LinearAutoBomberModel


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model = LinearAutoBomberModel(feature_extractor=lambda x: state_to_features(x))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # todo right now epsilon-greedy - change to softmax to avoid local maxima
    if self.train and random.random() < config.EPSILON:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(config.ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        return self.model.select_best_action(game_state, self)


def state_to_features(game_state: dict, weight_opponents_no_bomb=0.0) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :param weight_opponents_no_bomb: Float defining how much danger should be accounted for opponents
                                    without BOMB action available
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        # todo we need another representation for final state here!
        return np.random.rand(27)


    field_width, field_height = game_state['field'].shape
    assert field_width == field_height, "Field is not rectangular, some assumptions do not hold. Abort!"

    agent_position = np.asarray(game_state['self'][3], dtype='int')
    bombs_position = np.asarray([list(bomb[0]) for bomb in game_state['bombs']], dtype='int')
    bombs_countdown = np.asarray([bomb[1] for bomb in game_state['bombs']])
    explosions_position = np.argwhere(game_state['explosion_map'] > 0)
    coins_position = np.array(game_state['coins'], dtype='int')
    crates_position = np.argwhere(game_state['field'] == 1)
    opponents_position = np.asarray([list(player[3]) for player in game_state['others']], dtype='int')
    opponents_bomb_action = np.asarray([player[2] for player in game_state['others']])
    opponents_bomb_action = np.where(opponents_bomb_action, 1.0, weight_opponents_no_bomb)

    # TODO Evaluate normalization/scaling
    bomb_danger_zones = _compute_zones_heatmap(agent_position, bombs_position, lambda v, w: v ** (1 / w),
                                               bombs_countdown, lambda v: v / np.max(v))
    coins_zones = _compute_zones_heatmap(agent_position, coins_position, normalization_func=lambda v: v / np.max(v))
    crates_zones = _compute_zones_heatmap(agent_position, crates_position, normalization_func=lambda v: v / np.max(v))
    opponents_zones = _compute_zones_heatmap(agent_position, opponents_position, lambda v, w: v * w,
                                             opponents_bomb_action, lambda v: v / np.max(v))

    explosion_field_of_view = _object_in_field_of_view(agent_position, explosions_position, lambda v, w: v / w,
                                                       field_width)
    coins_field_of_view = _object_in_field_of_view(agent_position, coins_position, lambda v, w: v / w, field_width)
    crates_field_of_view = _object_in_field_of_view(agent_position, crates_position, lambda v, w: v / w, field_width)

    # TODO Set auxiliary reward for moving away from a danger zone

    features = np.concatenate((bomb_danger_zones, coins_zones, crates_zones, opponents_zones,
                               explosion_field_of_view, coins_field_of_view, crates_field_of_view), axis=None)

    return features


def _compute_zones_heatmap(agent_position, objects_position, weighting_func=None, weights=None,
                           normalization_func=None):
    """
    Computes the distance of given objects from the agent and determines their position relative to the agent.

    The game field is divided in 4 quadrants relative to the agent's position, each covering an angle of 90 degrees.
    The quadrants, i.e. zones, are thus above, left, below, right of the agent.

    An optional weighting can be applied to the objects.

    Parameters
    ----------
    agent_position : np.array
        Position of the agent (x, y)
    objects_position : np.array
        Position of the objects on the field
    weighting_func : callable, optional
        Function to additionally weight the objects
    weights : np.array
        Weights to apply to the objects' distance
    normalization_func : callable, optional
        Function to normalize (or scale) the aggregated value in the zones

    Returns
    -------
    list
        A list with 4 values (down, left, up, right) representing the (weighted) density
        of the specified objects in the quadrants around the agent
    """
    zones = np.zeros(shape=(4,))

    distances = np.linalg.norm(agent_position - objects_position, axis=1)
    if weighting_func:
        distances = weighting_func(distances, weights)
    angles = np.degrees(
        np.arctan2(objects_position[:, 1] - agent_position[1], objects_position[:, 0] - agent_position[0]))

    # TODO Evaluate if: map object to two zones if it is in-between
    # Computed: RIGHT; Actual: DOWN
    zones[0] = np.sum(
        distances[np.where(((angles >= 0) & (angles < 45)) | ((angles >= 315) | (angles <= 360)))])
    # Computed DOWN: Actual: LEFT
    zones[1] = np.sum(distances[np.where((angles >= 45) & (angles < 135))])
    # Computed LEFT: Actual: UP
    zones[2] = np.sum(distances[np.where((angles >= 135) & (angles < 225))])
    # Computed UP: Actual: RIGHT
    zones[3] = np.sum(distances[np.where((angles >= 225) & (angles < 315))])

    if normalization_func:
        zones = normalization_func(zones)

    return zones


def _object_in_field_of_view(agent_position, objects_position, normalization_func=None, norm_constant=None):
    """
    Specifies the field of view w.r.t the given objects.

    When computing the distance of the agent to the objects, the agent's own position
    is included, i.e. if the agent is ON the object the distance is 0.0 .

    Parameters
    ----------
    agent_position : np.array
        Position of the agent (x, y)
    objects_position : np.array
        Position of the objects on the field
    normalization_func : callable, optional
        Function to normalize (or scale) the distances on the 4 directions
    norm_constant :
        Constant used for the normalization

    Returns
    -------
    list
        A list with 4 values (down, left, up, right) representing the distance
        of the agent to the nearest object (if any) below, left, above, right of it.

    """
    # TODO Maybe scale values: small distance -> high value, high distance -> small value
    field_of_view = np.full(shape=(4,), fill_value=-1.)

    # Coordinate x is as of the framework field
    objects_on_x = np.where(objects_position[:, 0] == agent_position[0])
    # Directions are actual directions, i.e. after translation of framework fields
    objects_down = np.where(objects_position[objects_on_x, 1] >= agent_position[1])
    field_of_view[0] = np.linalg.norm(agent_position - objects_position[objects_down], axis=1).min()
    objects_up = np.where(objects_position[objects_on_x, 1] <= agent_position[1])
    field_of_view[2] = np.linalg.norm(agent_position - objects_position[objects_up], axis=1).min()

    # Coordinate y is as of the framework field
    objects_on_y = np.where(objects_position[:, 1] == agent_position[1])
    # Directions are actual directions, i.e. after translation of framework fields
    objects_left = np.where(objects_position[objects_on_y, 0] >= agent_position[0])
    field_of_view[1] = np.linalg.norm(agent_position - objects_position[objects_left], axis=1).min()
    objects_right = np.where(objects_position[objects_on_y, 0] <= agent_position[0])
    field_of_view[3] = np.linalg.norm(agent_position - objects_position[objects_right], axis=1).min()

    if normalization_func:
        field_of_view = normalization_func(field_of_view, norm_constant)

    return field_of_view

