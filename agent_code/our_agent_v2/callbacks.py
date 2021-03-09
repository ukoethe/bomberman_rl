import os
import pickle
import random

import numpy as np

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
        self.q_table = np.zeros((4*9*4, self.action_size))   #initi a q_table which has as many states as possible distances to coin possible
        #self.q_table = np.load("my-q-table_increase_featurespace-alpha=0.01.npy")
        
    else:
        self.logger.info("Loading model from saved state.")
        self.q_table = np.load("my-q-table_agentv13.npy")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    ########### (1) only allow valid actions: #############
    mask, VALIDE_ACTIONS, p =  get_valide_action(game_state)
    
    ########### (2) When in Training mode: #############
    # todo Exploration vs exploitation: take a decaying exploration rate
    if self.train:
        random_prob = self.epsilon 
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            execute_action = np.random.choice(VALIDE_ACTIONS)
            #print(VALIDE_ACTIONS, execute_action , p)
            return execute_action
        else:
            self.logger.debug("Choosing action from highes q_value.")
            
            # choose only from q_values which are valid actions: 
            q_values = self.q_table[state_to_features(game_state)][mask]
            execute_action = VALIDE_ACTIONS[np.argmax(q_values)]
            #print(VALIDE_ACTIONS , execute_action , p)
            return execute_action

    ########### (3) When in Game mode: #############
    else:
        
        random_prob = 0.1
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            execute_action = np.random.choice(VALIDE_ACTIONS)
            #print(VALIDE_ACTIONS, execute_action , p)
            return execute_action
        
        # choose only from q_values which are valid actions: 
        q_values = self.q_table[state_to_features(game_state)][mask]
        #print(q_values, state_to_features(game_state))
        execute_action = VALIDE_ACTIONS[np.argmax(q_values)]
        self.logger.debug("Querying model for action.")
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

    # (1) get direction to closest coin  #possible since reward when getting closer to coin
    coins_info = []
    for coin in coins:
        x_coin_dis = coin[0] - x
        y_coin_dis = coin[1] - y
        total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
        coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
        coins_info.append(coin_info)

    closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]
    
    # get direction to cloest coin, values possible for h,v = 0,-1,1 therefore 9 states
    if closest_coin_info[0] == 0:
        h = closest_coin_info[0]
    else: h = closest_coin_info[0]/np.abs(closest_coin_info[0])  # is -1 or 1 dependent on closest coin to left or right
        
    if closest_coin_info[1] == 0:
        v = closest_coin_info[1]
    else: v = closest_coin_info[1]/np.abs(closest_coin_info[1])  # is -1 or 1 dependent on closest coin to up or down
    
    # (2) whether horizontal distance is larger than vertical distance or not to prevent stuking 
  
    # hoizontal_distance_larger = 0
    # if abs(closest_coin_info[0]) > abs(closest_coin_info[1]):
    #     hoizontal_distance_larger = 1
    
    # (2) Check which direction is farest:
    
    if abs(closest_coin_info[0]) > abs(closest_coin_info[1]):
        if closest_coin_info[0] > 0:
            farest_distance = 0  # Right
        else: farest_distance = 1 # Left
    else:
        if closest_coin_info[1] > 0:
            farest_distance = 2  # Down
        else: farest_distance = 3 # Up
        
        
        
    # (3) encounter for relative postion of agent in arena: 
    # is between two invalide field horizontal (not L and R, do U and D) or not
    # is between two invalide field vertical (do L and R, not U and D) or not
    # will increase number of states with a factor 4
    mask, VALIDE_ACTIONS, p =  get_valide_action(game_state)
    
    relative_position_horizont = 0
    relative_position_vertical = 0
    
    if 'RIGHT' not in VALIDE_ACTIONS and 'LEFT' not in VALIDE_ACTIONS:
        relative_position_horizont = 1  # between_invalide_horizintal
    
    if 'UP' not in VALIDE_ACTIONS and 'DOWN' not in VALIDE_ACTIONS:
        relative_position_vertical = 1  # between_invalide_vertical
    
    features = [ h, v ,relative_position_horizont, relative_position_vertical , farest_distance ]
    
    # do encoding
    grid = np.zeros((3,3,2,2,4))
    l = 0
    for i in range (len(grid)):
        for j in range (len(grid[0])):
            for k in range (len(grid[0][0])):
                for m in range (len(grid[0][0][0])):
                    for n in range (len(grid[0][0][0][0])):
                        grid[i,j,k,m,n] = l
                        l+=1
    state_number = int(grid[int(h+1),int(v+1),relative_position_horizont, relative_position_vertical, farest_distance])
    
    return state_number

def state_to_features_old(game_state: dict) -> np.array:
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

    # (1) get relative step distances to closest coin as one auto hot encoder
    coins_info = []
    for coin in coins:
        x_coin_dis = coin[0] - x
        y_coin_dis = coin[1] - y
        total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
        coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
        coins_info.append(coin_info)

    closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]

    h = closest_coin_info[0] + max_distance_x  
    v = closest_coin_info[1] + max_distance_y 
    
    # (2) encounter for relative postion of agent in arena: 
    # is between two invalide field horizontal (not L and R, do U and D)
    # is between two invalide field vertical (do L and R, not U and D)
    # somewhere else (not L and R, not U and D)
    # will increase number of states with a factor 3
    mask, VALIDE_ACTIONS, p =  get_valide_action(game_state)
    
    relative_position = 0
    
    if 'RIGHT' not in VALIDE_ACTIONS and 'LEFT' not in VALIDE_ACTIONS:
        relative_position = 1  # between_invalide_horizintal
    
    if 'UP' not in VALIDE_ACTIONS and 'DOWN' not in VALIDE_ACTIONS:
        relative_position = 2  # between_invalide_vertical
    
    # do encoding
    grid = np.zeros((2*(s.COLS-2),2*(s.ROWS-2), 3))
    l = 0
    for i in range (len(grid)):
        for j in range (len(grid[0])):
            for k in range (len(grid[0][0])):
                grid[i,j,k] = l
                l+=1
    
    state_number = int(grid[h,v,relative_position])
    #print(grid)
    #print(closest_coin_info[0], closest_coin_info[1],relative_position, state_number )
    
    return state_number


def get_valide_action(game_state: dict):
    """
    Given the gamestate, check which actions are valide.

    :param game_state:  A dictionary describing the current game board.
    :return: mask which ACTIONS executable
             list of VALIDE_ACTIONS
             uniform random distribution for VALID_ACTIONS
    """

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']
    
    # check for valid actions
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (game_state['explosion_map'][d] <= 1) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x , y - 1) in valid_tiles: valid_actions.append(1) # UP
    else: valid_actions.append(0)
    if (x + 1, y) in valid_tiles: valid_actions.append(1) # RIGHT
    else: valid_actions.append(0)   
    if (x, y + 1) in valid_tiles: valid_actions.append(1) # DOWN
    else: valid_actions.append(0)
    if (x -1 , y ) in valid_tiles: valid_actions.append(1) # LEFT
    else: valid_actions.append(0)
    if (x, y) in valid_tiles: valid_actions.append(1) # WAIT
    else: valid_actions.append(0)
    if (bombs_left > 0) : valid_actions.append(0)  # BOMB drop bomb alwas impossible
    else: valid_actions.append(0)
    
    #create mask which only allows valid move
    mask = (np.array(valid_actions)==1)
    VALIDE_ACTIONS = np.array(ACTIONS)[mask]
    p = np.random.dirichlet(np.ones(len(VALIDE_ACTIONS)),size=1)[0] 
    
    return mask, VALIDE_ACTIONS, p
