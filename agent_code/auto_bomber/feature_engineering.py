import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        # todo we need another representation for final state here!
        return np.random.rand(21)

    field_width, field_height = game_state['field'].shape
    assert field_width == field_height, "Field is not rectangular, some assumptions do not hold. Abort!"

    agent_position = np.asarray(game_state['self'][3], dtype='int')
    agent_bomb_action = np.asarray(game_state['self'][2], dtype='int')
    bombs_position = np.atleast_2d(np.asarray([list(bomb[0]) for bomb in game_state['bombs']], dtype='int'))
    explosions_position = np.argwhere(game_state['explosion_map'] > 0)
    coins_position = np.atleast_2d(np.array(game_state['coins'], dtype='int'))
    relevant_coins_position = coins_position[~np.isin(coins_position, agent_position).all(axis=1)]
    crates_position = np.argwhere(game_state['field'] == 1)
    walls_position = np.argwhere(game_state['field'] == -1)
    weight_opponents_with_bomb = 0.8
    opponents_position = np.atleast_2d(np.asarray([list(player[3]) for player in game_state['others']], dtype='int'))
    opponents_bomb_action = np.asarray([player[2] for player in game_state['others']])
    opponents_bomb_action = np.where(opponents_bomb_action, weight_opponents_with_bomb, 1.0)

    #############################################
    #                 DISCARDED                 #
    # Bombs zones logic:                        #
    # Due to bad performance in empirical tests #
    #############################################
    #
    #
    # bombs_zones = _compute_zones_heatmap(agent_position, bombs_position, 0.0,
    #                                      lambda v, w: np.where(v > 0., v[(3 + w) - v >= 0] ** w[(3 + w) - v >= 0], 0.0),
    #                                      bombs_countdown,
    #                                      lambda v: np.mean(v[v != 0.0]) if v[v != 0.0].size != 0 else 0.0,
    #                                      lambda v: -1 * np.divide(1, v, out=np.zeros_like(v), where=v != 0))

    coins_zones = _compute_zones_heatmap(agent_position, relevant_coins_position, 0.0,
                                         aggregation_func=lambda v: np.mean(v) if v.size != 0 else 0.0,
                                         normalization_func=lambda v: np.divide(1, v, out=np.zeros_like(v), where=v != 0))
    crates_zones = _compute_zones_heatmap(agent_position, crates_position, 0.0,
                                          aggregation_func=lambda v: np.mean(v) if v.size != 0 else 0.0,
                                          normalization_func=lambda v: np.divide(1, v, out=np.zeros_like(v), where=v != 0))
    opponents_zones = _compute_zones_heatmap(agent_position, opponents_position, 0.0,
                                             weighting_func=lambda v, w: v * w,
                                             weights=opponents_bomb_action,
                                             aggregation_func=lambda v: np.mean(v) if v.size != 0 else 0.0,
                                             normalization_func=lambda v: np.divide(1, v, out=np.zeros_like(v), where=v != 0))

    bombs_field_of_view = _object_in_field_of_view(agent_position, bombs_position, 0.0,
                                                   lambda v, w: -1 * np.divide(1, v, out=np.zeros_like(v),
                                                                               where=v != 0),
                                                   None)
    explosion_field_of_view = _object_in_field_of_view(agent_position, explosions_position, 0.0)
    coins_field_of_view = _object_in_field_of_view(agent_position, relevant_coins_position, 0.0,
                                                   lambda v, w: np.divide(1, v, out=np.zeros_like(v), where=v != 0),
                                                   None)
    crates_field_of_view = _object_in_field_of_view(agent_position, crates_position, 0.0,
                                                    lambda v, w: np.divide(1, v, out=np.zeros_like(v), where=v != 0),
                                                    None)
    opponents_field_of_view = _object_in_field_of_view(agent_position, opponents_position, 0.0,
                                                        lambda v, w: np.divide(1, v, out=np.zeros_like(v), where=v != 0),
                                                        None)
    walls_field_of_view = _object_in_field_of_view(agent_position, walls_position, 0.0)

    f_obstacles = np.zeros((4,))
    f_obstacles[walls_field_of_view == 1.] = -1.
    f_obstacles[explosion_field_of_view == 1.] = -1.
    f_obstacles[bombs_field_of_view == -1.] = -1.
    f_obstacles[crates_field_of_view == 1.] = -1.

    new_bombs_field_of_view = np.copy(bombs_field_of_view)
    if bombs_position.size != 0 and np.isin(bombs_position, agent_position).all(axis=1).any():
        new_bombs_field_of_view = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        for i in range(4):
            new_bombs_field_of_view[i] += np.sum(-1 * np.delete(bombs_field_of_view / 3., i))
    new_bombs_field_of_view[bombs_field_of_view == -1.] = -1.
    f_bombs = new_bombs_field_of_view

    f_coins = np.sum(np.vstack((coins_zones, 5 * coins_field_of_view)), axis=0)
    f_coins[walls_field_of_view == 1.] = 0.
    if not np.all((f_coins == 0.)):
        f_coins = np.where(f_coins == 0., -np.inf, f_coins)
        f_coins = softmax(f_coins)

    f_crates = np.sum(np.vstack((crates_zones, 5 * crates_field_of_view)), axis=0)
    f_crates[walls_field_of_view == 1.] = 0.
    if not np.all((f_crates == 0.)):
        f_crates = np.where(f_crates == 0., -np.inf, f_crates)
        f_crates = softmax(f_crates)
    f_crates[crates_field_of_view == 1.] = -1.

    f_opponents = np.sum(np.vstack((opponents_zones, 5 * opponents_field_of_view)), axis=0)
    f_opponents[walls_field_of_view == 1.] = 0.
    if not np.all((f_opponents == 0.)):
        f_opponents = np.where(f_opponents == 0., -np.inf, f_opponents)
        f_opponents = softmax(f_opponents)
    f_opponents[opponents_field_of_view == 1.] = -1.

    features = np.concatenate((f_coins, f_crates, f_bombs, f_opponents, f_obstacles, agent_bomb_action), axis=None)

    return features


def _compute_zones_heatmap(agent_position, objects_position, initial, weighting_func=None, weights=None,
                           aggregation_func=None, normalization_func=None):
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
        A list with 4 values (right, down, left, up) representing the (weighted) density
        of the specified objects in the quadrants around the agent
    """
    zones = np.full(shape=(4,), fill_value=initial)

    if objects_position.size == 0:
        return zones

    agent_position = np.atleast_2d(agent_position)
    distances = cdist(agent_position, objects_position, 'cityblock').squeeze(axis=0)
    agent_position = agent_position[0]
    if weighting_func:
        distances = weighting_func(distances, weights)
    angles = np.degrees(
        np.arctan2(objects_position[:, 1] - agent_position[1], objects_position[:, 0] - agent_position[0]))
    angles = (angles + 360) % 360

    # TODO Evaluate if: map object to two zones if it is in-between
    # Computed: RIGHT; Actual: RIGHT
    zones[0] = aggregation_func(
        distances[np.where(((angles >= 0) & (angles < 45)) | ((angles >= 315) & (angles <= 360)))])
    # Computed: UP; Actual: DOWN
    zones[1] = aggregation_func(distances[np.where((angles >= 45) & (angles < 135))])
    # Computed: LEFT; Actual: LEFT
    zones[2] = aggregation_func(distances[np.where((angles >= 135) & (angles < 225))])
    # Computed: DOWN; Actual: UP
    zones[3] = aggregation_func(distances[np.where((angles >= 225) & (angles < 315))])

    if normalization_func:
        zones = normalization_func(zones)

    return zones


def _object_in_field_of_view(agent_position, objects_position, initial, normalization_func=None, norm_constant=None):
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
        A list with 4 values (right, down, left, up) representing the distance
        of the agent to the nearest object (if any) below, left, above, right of it.

    """
    # TODO Maybe scale values: small distance -> high value, high distance -> small value
    field_of_view = np.full(shape=(4,), fill_value=initial)

    if objects_position.size == 0:
        return field_of_view

    agent_position = np.atleast_2d(agent_position)

    # Coordinate x is as of the framework field
    objects_on_x = objects_position[np.where(objects_position[:, 0] == agent_position[0, 0])]
    # Directions are actual directions, i.e. after translation of framework fields
    objects_down = objects_on_x[np.where(objects_on_x[:, 1] >= agent_position[0, 1])]
    if not objects_down.size == 0:
        field_of_view[1] = cdist(agent_position, objects_down, 'cityblock').squeeze(axis=0).min()
    objects_up = objects_on_x[np.where(objects_on_x[:, 1] <= agent_position[0, 1])]
    if not objects_up.size == 0:
        field_of_view[3] = cdist(agent_position, objects_up, 'cityblock').squeeze(axis=0).min()

    # Coordinate y is as of the framework field
    objects_on_y = objects_position[np.where(objects_position[:, 1] == agent_position[0, 1])]
    # Directions are actual directions, i.e. after translation of framework fields
    objects_right = objects_on_y[np.where(objects_on_y[:, 0] >= agent_position[0, 0])]
    if not objects_right.size == 0:
        field_of_view[0] = cdist(agent_position, objects_right, 'cityblock').squeeze(axis=0).min()
    objects_left = objects_on_y[np.where(objects_on_y[:, 0] <= agent_position[0, 0])]
    if not objects_left.size == 0:
        field_of_view[2] = cdist(agent_position, objects_left, 'cityblock').squeeze(axis=0).min()

    if normalization_func:
        field_of_view = normalization_func(field_of_view, norm_constant)

    return field_of_view
