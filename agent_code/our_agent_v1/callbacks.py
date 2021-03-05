import os
import pickle
import random
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import settings as s
from operator import itemgetter


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.96


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
    self.action_size = len(ACTIONS) #Get size of the action
        
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        self.model = KNeighborsRegressor(n_jobs=-1)
        # self.model = MultiOutputRegressor(SVR(), n_jobs=8)
        self.isFit = False
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """    
    # (5) valid actions
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = game_state['explosion_map']
    
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (game_state['explosion_map'][d] <= 1) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x , y + 1) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x + 1, y) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)   
    if (x, y - 1) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x -1 , y ) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x, y) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (bombs_left > 0) : valid_actions.append(1)
    else: valid_actions.append(0)
    
    mask = valid_actions
    msk = (np.array(mask)==1)
    VALIDE_ACTIONS = np.array(ACTIONS)[msk]
    p = np.random.dirichlet(np.ones(len(VALIDE_ACTIONS)),size=1)[0] 
    #print(VALIDE_ACTIONS)
    
    # todo Exploration vs exploitation
    if self.train:
        random_prob = self.epsilon
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            #print("random choice epsilon greedy" , np.random.choice(VALIDE_ACTIONS, p=p))
            return np.random.choice(VALIDE_ACTIONS, p=p)
        
        if self.isFit == True:
            q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))
            # choose only from q_values which are valid actions: 
            mask_arr = np.ma.masked_array(q_values[0], mask = ~msk)
            # applying MaskedArray.argmax methods to mask array 
            execute_action = ACTIONS[mask_arr.argmax()]
            #print("max q value choice " , execute_action)
            
        else:
            q_values = np.zeros(self.action_size).reshape(1, -1) 
            execute_action = np.random.choice(VALIDE_ACTIONS, p=p)
            #print("not training yet- choice ", execute_action)

        self.logger.debug("Querying model for action.") 
        return execute_action
    
    else:
        random_prob = .01
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])
        
        q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))
        # choose only from q_values which are valid actions: 
        mask_arr = np.ma.masked_array(q_values[0], mask = ~msk)
        # applying MaskedArray.argmax methods to mask array 
        execute_action = ACTIONS[mask_arr.argmax()]
        return execute_action
        


def state_to_features( game_state: dict) -> np.array:
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
        
    # break down state into one image (feature possibility A):
    Y = arena
    Y[x,y] = 50
    for coin in coins:
        Y[coin] = 10
    for bomb in bombs:
        Y[bomb[0]] = -10*(bomb[1]+1)
    np.where(bomb_map != 0, Y, -10)
    
    Y = Y.reshape(1, -1)
    
    # break down into the follwoing features (feature possibility B):
    ''' 
        ['distance_agent_to_center_lr', 'distance_agent_to_center_ud', 'total_distance_center',
        'steps_to_closest_coin_lr', 'steps_to_closest_coin_ud', 'total_distance_closest_coin',
        'steps_to_second_closest_coin_lr', 'steps_to_second_closest_coin_ud', 'total_distance_second_closest_coin',
        ,.... , 
        'steps_to_farest_coin_lr', 'steps_to_farest_coin_ud' ,'total_distance_farest_coin',
        'steps_to_bomb1_lr', 'steps_to_bomb1_coin_ud', 'timer_bomb1',
        ,...,
        'steps_to_bomb4_coin_lr', 'steps_to_bomb4_coin_ud' , 'timer_bomb4',      
        'LEFT_valid', 'RIGHT_valid', 'UP_valid' ,'DOWN_valid', 'WAIT_valid', BOMB_valid',
        'dead_zone_yes_no'] 
    '''
    # if 'steps_to_closest_coin_lr' is negativ -> means to left, positive means to right
    # if action is vaild 1 otherwise 0
    
    # (1) center agent:
    x_cen = x-(s.COLS-2)/2
    y_cen = y-(s.ROWS-2)/2
    total_step_distance = abs(x_cen) + abs(y_cen)
    agent_info = (x_cen, x_cen, total_step_distance)
    
    # (2) step difference to coins:
    # take as given that there are 9 coins in total
    
    coins_info = []
    for coin in coins:
        x_rel_coin = coin[0] - x
        y_rel_coin = coin[1] - y
        total_step_distance = abs(x_rel_coin) + abs(y_rel_coin)
        coin_info = (x_rel_coin, y_rel_coin, total_step_distance)
        coins_info.append(coin_info)
    while len(coins_info) < 9:
        coins_info.append((99,99,99))
    coins_info = sorted(coins_info, key=itemgetter(2))
    
    # (4) bomb_distance, dead zone:
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    if bomb_map[x,y] != 5:
        dead_zone = 1
    else: dead_zone = 0
        
    bombs_info = []    
    for bomb in bombs:
        x_rel_bomb = bomb[0][0] - x
        y_rel_bomb = bomb[0][1] - y
        timer = bomb[1]
        bomb_info = (x_rel_bomb, y_rel_bomb, timer )
        bombs_info.append(bomb_info)
    while len(bombs_info) < 4:
        bombs_info.append((99,99,99))
    bombs_info = sorted(bombs_info, key=itemgetter(2))

    
    # For example, you could construct several channels of equal shape, ...
    channels = []
   
    channels.append(agent_info)
    #for coin_info in coins_info:
    #    channels.append(coin_info)
    # only closest coin distance:
    channels.append(coins_info[0])
    
    # only closest bomb distance:
    #for bomb_info in bombs_info:
    #    channels.append(bomb_info)
    channels.append(bombs_info[0])
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    
    # and return them as a vector    
    X = np.append(stacked_channels.reshape(-1), dead_zone)
    
    #print(X)
    return X
