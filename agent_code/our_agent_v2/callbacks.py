import os
import pickle
import random
from operator import itemgetter

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA

import settings as s
import events as e
from .callbacks import DR_BATCH_SIZE

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# ---------------- Parameters ----------------
FILENAME = "SGD_agent_v1"         # Filename of for model output (excl. extension).
ACT_STRATEGY = 'softmax'          # Options: 'softmax', 'eps-greedy'
# --------------------------------------------

fname = f"{FILENAME}.pt" # Adding the file extension.

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
    # Save constants.
    self.action_size = len(ACTIONS)
    self.actions = ACTIONS

    # Assign decision strategy.
    self.act_strategy = ACT_STRATEGY

    # Set up Incremental PCA for feature reduction.
    n_comp = 100
    self.transformer = IncrementalPCA(n_components=n_comp, batch_size=DR_BATCH_SIZE) 

    # Setting up the model.
    if os.path.isfile(fname):
        self.logger.info("Loading model from saved state.")
        with open(fname, "rb") as file:
            self.model = pickle.load(file)
        self.is_fitted = True
    elif self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = MultiOutputRegressor(SGDRegressor(alpha=0.0001, warm_start=True))
        self.is_fitted = False
    else:
        raise ValueError(f"Could not locate saved model {fname}")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # --------- (1) Only allow valid actions: -----------------
    mask, valid_actions =  get_valid_action(game_state)
    
    # --------- (2a) Softmax decision strategy: ---------------
    if self.act_strategy == 'softmax':
        # Softmax temperature. During training, we anneal the temperature. In 
        # game mode, we use a predefined (optimal) temperature. Limiting cases:
        # tau -> 0 : a = argmax Q(s,a) | tau -> +inf : uniform prob dist P(a).
        if self.train:
            tau = self.tau
        else:
            tau = 0.1 # TODO: Hyper-parameter which needs optimization.
        if self.is_fitted:
            self.logger.debug("Choosing action from softmax distribution.")
            # Q-values for the current state.
            q_values = self.model.predict(state_to_features(game_state))[0][mask]
            # Normalization for numerical stability.
            qtau = q_values/tau - np.max(q_values/tau)
            # Probabilities from Softmax function.
            p = np.exp(qtau) / np.sum(np.exp(qtau))        
        else:
            # Uniformly random action when Q not yet initialized.
            self.logger.debug("Choosing action uniformly at random.")
            p = np.ones(len(valid_actions))/len(valid_actions)
        # Pick choice from valid actions with the given probabilities.
        return np.random.choice(valid_actions, p=p)

    # --------- (2b) Epsilon-Greedy decision strategy: --------
    elif self.act_strategy == 'eps-greedy':
        if self.train:
            random_prob = self.epsilon
        else:
            random_prob = 0.1 # TODO: Hyper-parameter which needs optimization.
        if random.random() < random_prob or not self.is_fitted:
            self.logger.debug("Choosing action uniformly at random.")
            execute_action = np.random.choice(valid_actions)
        else:
            self.logger.debug("Choosing action with highest q_value.")
            q_values = self.model.predict(state_to_features(game_state))[0][mask]
            execute_action = valid_actions[np.argmax(q_values)]
        return execute_action
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")


def state_to_vect():
    return vect


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

    # #TODO:FIX THIS
    h = 0
    v = 0
    if coins_info:
        closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]  # TODO: This breaks with no coins.
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
    mask, valid_actions =  get_valid_action(game_state)
    
    relative_position_vertical = 0
    relative_position_horizontal = 0
    
    if 'RIGHT' not in valid_actions and 'LEFT' not in valid_actions:
        relative_position_horizontal = 1
    
    if 'UP' not in valid_actions and 'DOWN' not in valid_actions:
        relative_position_vertical = 1
    
    features = np.array([h , v , relative_position_horizontal , relative_position_vertical])

    return features.reshape(1, -1)


def get_valid_action(game_state: dict):
    """
    Given the gamestate, check which actions are valide.

    :param game_state:  A dictionary describing the current game board.
    :return: mask which ACTIONS are executable
             list of VALID_ACTIONS
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

    return mask, valid_actions
