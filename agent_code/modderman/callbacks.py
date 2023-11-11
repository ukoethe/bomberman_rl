import os
import pickle
import random

import numpy as np
from .features_actual import BombermanFeatures
from sklearn.tree import DecisionTreeRegressor


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_FEATURES = 19

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
    self.feature_extractor = BombermanFeatures()

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.decision_trees = [DecisionTreeRegressor() for i in range(len(ACTIONS))]

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.decision_trees = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation using epsilon-greedy
    random_prob = .1

    if self.train and (random.random() < random_prob or game_state['round'] < 50): # Choose random actions for first 50 rounds
        self.logger.debug("Choosing action purely at random.")
        #80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    
    #approximate Q by decision trees trained on each action
    Q =  [self.decision_trees[action_idx_to_test].predict([state_to_features(self, game_state)]) for action_idx_to_test in range(len(ACTIONS))]

    # Take the action with maximum value approximated from our decision trees
    action_idx = np.argmax(Q)

    return ACTIONS[action_idx]

def state_to_features(self, game_state: dict) -> np.array:
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
    

    return self.feature_extractor.state_to_features(game_state)

    # features = np.zeros(NUM_FEATURES)

    # field = np.array(game_state['field'])
    # coins = np.array(game_state['coins'])
    # bombs = game_state['bombs']
    # explosion_map = np.array(game_state['explosion_map'])
    # _, _, own_bomb, (x, y) = game_state['self']
    # others = game_state['others']
    # others_position = np.zeros( (len(others), 2), dtype=int)

    # for i,opponent in enumerate(others):
    #     others_position[i] = opponent[3]

    # explosion_indices = np.array(np.where(explosion_map > 0)).T

    # coin_distance = np.inf
    # crate_distance = np.inf
    # opponent_distance = np.inf



# def state_to_features(game_state: dict) -> np.array:
#     # Initialize the feature vector
#     features = []

#     # Extract relevant information from the game state
#     field = game_state['field']
#     bombs = game_state['bombs']
#     explosion_map = game_state['explosion_map']
#     coins = game_state['coins']
#     self_info = game_state['self']
#     self_x, self_y = self_info[3]

#     # 1. Encode the game board (field)
#     field_encoded = np.copy(field)
#     field_encoded[field_encoded == 1] = -1  # Encode crates as -1
#     field_encoded[field_encoded == 0] = 1   # Encode free tiles as 1
#     # You may choose to encode stone walls differently if needed.

#     # 2. Encode bomb information
#     # Calculate distances to the nearest bomb(s) and their countdowns
#     min_bomb_distance = np.inf
#     min_bomb_countdown = np.inf
#     for (x, y), countdown in bombs:
#         dist = np.abs(self_x - x) + np.abs(self_y - y)
#         min_bomb_distance = min(min_bomb_distance, dist)
#         min_bomb_countdown = min(min_bomb_countdown, countdown)

#     # Append bomb-related features
#     features.extend([min_bomb_distance, min_bomb_countdown])

#     # 3. Encode explosion map
#     # Calculate the maximum explosion duration in the agent's vicinity
#     max_explosion_duration = np.max(explosion_map[self_x - 1:self_x + 2, self_y - 1:self_y + 2])

#     # Append explosion-related features
#     features.append(max_explosion_duration)

#     # 4. Encode coin information
#     # Calculate the distance to the nearest coin
#     min_coin_distance = np.inf
#     for coin_x, coin_y in coins:
#         dist = np.abs(self_x - coin_x) + np.abs(self_y - coin_y)
#         min_coin_distance = min(min_coin_distance, dist)

#     # Append coin-related features
#     features.append(min_coin_distance)

#     # 5. Additional features (e.g., self score, availability of the 'BOMB' action)
#     features.append(self_info[1])  # Self score
#     features.append(int(self_info[2]))  # Availability of 'BOMB' action

#     # Convert the feature vector to a numpy array
#     features = np.array(features, dtype=np.float32)

#     return features.reshape(-1)
