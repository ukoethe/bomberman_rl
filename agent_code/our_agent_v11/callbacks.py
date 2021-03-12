import os
import pickle
import random
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import settings as s
from operator import itemgetter
from sklearn.linear_model import LinearRegression, SGDRegressor
#from sklearn.preprocessing import PolynomialFeatures


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
        
    if self.train:
        self.logger.info("Setting up model from scratch.")
        #self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        #self.model = KNeighborsRegressor(n_jobs=-1)
        # self.model = MultiOutputRegressor(SVR(), n_jobs=8)
        self.isFit = False
        #self.model = LinearRegression()
        #self.model = MultiOutputRegressor(SGDRegressor( alpha = LEARNING_RATE ))
        self.q_table = np.zeros((4*((s.COLS-2)*(s.ROWS-2)), self.action_size))
        
    else:
        self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
        #    self.model = pickle.load(file)
        self.q_table = np.load("my-q-table-longer.npy")


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
    if (x , y - 1) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x + 1, y) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)   
    if (x, y + 1) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x -1 , y ) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (x, y) in valid_tiles: valid_actions.append(1)
    else: valid_actions.append(0)
    if (bombs_left > 0) : valid_actions.append(0)  # drop bomb alwas impossible
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
            execute_action = np.random.choice(VALIDE_ACTIONS, p=p)
            #print ("random choice epsilon greedy" , VALIDE_ACTIONS, execute_action)
            return execute_action
        
        if self.isFit == True:
            #q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))
            #mask_arr = np.ma.masked_array(q_values[0], mask = ~msk)
            
            q_values = self.q_table[state_to_features(game_state)][msk]
            execute_action = VALIDE_ACTIONS[np.argmax(q_values)]
            #print("max q value choice - options 2 " , execute_action)
            
        else:
            q_values = np.zeros(self.action_size).reshape(1, -1) 
            execute_action = np.random.choice(VALIDE_ACTIONS, p=p)
            #print("not training yet- choice ", execute_action)
            
        #print ("max q value choice " , VALIDE_ACTIONS, execute_action)
        self.logger.debug("Querying model for action.") 
        return execute_action
    
    else:
        random_prob = .01
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0])
        
        # q_values = self.model.predict(state_to_features(game_state).reshape(1, -1))
        # choose only from q_values which are valid actions: 
        # mask_arr = np.ma.masked_array(q_values[0], mask = ~msk)

        q_values = self.q_table[state_to_features(game_state)]
        mask_arr = np.ma.masked_array(q_values, mask = ~msk)
                
        print (q_values)
        # applying MaskedArray.argmax methods to mask array 
        execute_action = ACTIONS[mask_arr.argmax()]
        print ("max q value choice " , VALIDE_ACTIONS, execute_action)
        
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
    
    max_distance_x = s.ROWS - 2
    max_distance_y = s.COLS - 2
    
    # get relative step distances to closest coin as one auto hot encoder
    coins_info = []
    for coin in coins:
        x_coin_dis = coin[0] - x
        y_coin_dis = coin[1] - y
        total_step_distance = abs(x_coin_dis) + abs(y_coin_dis)
        coin_info = (x_coin_dis , y_coin_dis , total_step_distance)
        coins_info.append(coin_info)
    #while len(coins_info) < 9:
    #    coins_info.append((99,99,99))
    closest_coin_info = sorted(coins_info, key=itemgetter(2))[0]
    
    #print("The relative distance to the closest coin is: ", closest_coin_info[0], closest_coin_info[1])
    h = closest_coin_info[0] + max_distance_x  
    v = closest_coin_info[1] + max_distance_y 
    
    # do encoding
    grid = np.zeros((2*(s.COLS-2),2*(s.ROWS-2)))
    l = 0
    for i in range (len(grid)):
        for j in range (len(grid[0])):
            grid[i,j] = l
            l+=1
    
    X = grid[h,v] # will be rows in q_table
    # each state ( of closest coin) becomes one specific number (entry in q table)
    # create grid (17,17) with entry 0 - 288
    # take value from [h,v] position as X
    return int(X)
