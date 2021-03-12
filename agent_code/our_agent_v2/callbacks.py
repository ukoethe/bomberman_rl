import os
import pickle
import random

import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor

import settings as s
import events as e
from operator import itemgetter

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    self.action_size = len(ACTIONS)
    self.actions = ACTIONS 
        
    if self.train:
        self.logger.info("Setting up model from scratch.")
        #self.q_table = np.zeros((3*4*((s.COLS-2)*(s.ROWS-2)), self.action_size))   #initi a q_table which has as many states as possible distances to coin possible
        #self.q_table = np.load("my-q-table_increase_featurespace-alpha=0.01.npy")
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.0001))
        
    else:
        self.logger.info("Loading model from saved state.")
        #self.q_table = np.load("my-q-table_agentv12_1coin.npy")
        with open("my-q-learning_Mulit_SGD_agentv12.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    ########### (1) only allow valid actions: #############
    mask, valid_actions, p =  get_valid_action(game_state)
    
    ########### (2) When in Training mode: #############
    # todo Exploration vs exploitation: take a decaying exploration rate
    if self.train:
        random_prob = self.epsilon 
        if random.random() < random_prob or self.is_init:
            # Uniformly & randomly picking a action from subset of valid actions.
            self.logger.debug("Choosing action purely at random.")
            execute_action = np.random.choice(valid_actions)
        else:
            # Choose action with maximum Q-value from subset of valid actions.
            self.logger.debug("Choosing action from highes q_value.")
            q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))[0][mask]
            execute_action = valid_actions[np.argmax(q_values)]

    ########### (3) When in Game mode: #############
    else:
        random_prob = 0.1
        if random.random() < random_prob:
            # Uniformly & randomly picking a action from subset of valid actions.
            self.logger.debug("Choosing action purely at random.")
            execute_action = np.random.choice(valid_actions)
        else:
            # Choose action with maximum Q-value from subset of valid actions.
            self.logger.debug("Querying model for action.")
            q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))[0][mask]
            execute_action = valid_actions[np.argmax(q_values)]
    
    return execute_action


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
    
    # Make a simple encoder to give every distance to closest coin a own state number :
    # e.g agent position (x,y) = (1,1) and closest coin position (1,5) -> distance (0 , 4) state number 4  
    # e.g agent position (x,y) = (15,15) and closest coin position (1,1) -> distance (-14 , -14) state number 4 = 23 
    # e.g agent position (x,y) = (1,1) and closest coin position (15,15) -> distance (14 , 14) state number 4 = 23 
    
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']    
    
    max_distance_x = s.ROWS - 2 #s.ROWS - 3 ? 
    max_distance_y = s.COLS - 2

    # (1) get relative,normlaized step distances to closest coin
    coins_info = []
    for coin in coins:
        x_coin_dis = coin[0] - x
        y_coin_dis = coin[1] - y
        total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
        coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
        coins_info.append(coin_info)

    closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]
    #print(closest_coin_info)
    if closest_coin_info[2] == 0:
        h = 0
        v = 0
    else:
        h = closest_coin_info[0]/closest_coin_info[2]  #normalize with total difference to coin   
        v = closest_coin_info[1]/closest_coin_info[2]  

    # (2) encounter for relative postion of agent in arena: 
    # is between two invalide field horizontal (not L and R, do U and D)
    # is between two invalide field vertical (do L and R, not U and D)
    # somewhere else (not L and R, not U and D)
    # will increase number of states with a factor 3
    mask, valid_actions, p =  get_valid_action(game_state)
    
    relative_position_vertical = 0
    relative_position_horizintal = 0
    
    if 'RIGHT' not in valid_actions and 'LEFT' not in valid_actions:
        relative_position_horizintal = 1  # between_invalide_horizintal
    
    if 'UP' not in valid_actions and 'DOWN' not in valid_actions:
        relative_position_vertical = 1  # between_invalide_vertical
    
    features = np.array([h , v , relative_position_horizintal , relative_position_vertical])
    # print(features.reshape(-1))
    return features.reshape(-1)


def get_valid_action(game_state: dict):
    """
    Given the gamestate, check which actions are valide.

    :param game_state:  A dictionary describing the current game board.
    :return: mask which ACTIONS are executable
             list of VALID_ACTIONS
             uniform random distribution for VALID_ACTIONS
    """
    aggressive_play = True # Allow agent to drop bombs. 

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']
    
    # Check for valid actions.
    #            ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    valid_actions = []
    mask = np.zeros(len(ACTIONS))

    # Movement:
    for i, d in enumerate(directions):
        if ((arena[d] == 0)    and # Is a free tile
            (bomb_map[d] <= 1) and # No ongoing explosion
            (not d in others)  and # Not occupied by other player
            (not d in bomb_xys)):  # No bomb placed

            valid_actions.append(ACTIONS[i]) # Append the valid action.
            mask[i] = 1                      # Binary mask
            
    # Bombing:
    if bombs_left and aggressive_play: 
        valid_actions.append(ACTIONS[-1])
        mask[-1] = 1

    # Convert binary mask to boolean mask of the valid moves.
    mask = (mask == 1)
    
    # Convert list to numpy array (# TODO Is this neccesary?)
    valid_actions = np.array(valid_actions)

    # Corresponding probabilites from Dirichlet dist.
    p = np.random.dirichlet(np.ones(len(valid_actions)), size=1)[0] 
    
    return mask, valid_actions, p
