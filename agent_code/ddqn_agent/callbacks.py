import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        
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
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    """
    Documentation of what you have on this point:
    game_state:
        'field' 17x17 array with 0=free, -1=undestroyable, 1=crates
        'self' (name, 0, False, Position)
        'others' (...)
        'bombs' List of positions [((X,Y), Turns), ((...),...), ...]
        'coins' List of positions [(X, Y), ...]
        'explosion_map' 17x17 array 0=no explosion 1=explosion

    Transitions:
        Deque with Action, Next State, Reward, State
    """
    state_to_features(game_state)
    return np.random.choice(ACTIONS, p=self.model)


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
    
    #IDEAS
    # Take relative positions
    # Only take window around the agent
    # Handle explosions like unbreakable walls and punish invalid moves all the same (combine explosion and normal field)
    game_state_position = {
        'others' : [other[3] for other in game_state["others"]],
        'coins' : game_state['coins'],
        'bombs' : game_state['bombs']
    }
    self_pos = game_state["self"][3]

    game_state_position = {key:relative_position(items, game_state['self'][3]) for key, items in game_state_position.items()}

    # Only take n tiles above/below/right/left of the agent
    vr = 2 # vision range - The problem with a vr > 2 is that at the outside walls you need to artifically add fields so the board is always the same size
    
    row = game_state['field'][self_pos[0], max(0, self_pos[1] - vr):min(17, self_pos[1] +vr+1)]
    col = game_state['field'][max(0, self_pos[0] - vr):min(17, self_pos[0] +vr+1), self_pos[1]]
    
    # This pads the window when the agent is next to the outer walls
    if self_pos[1] + vr > 16:
        row = np.insert(row, vr*2, -1)
    elif self_pos[1] - vr < 0:
        row = np.insert(row, 0, -1)
    if self_pos[0] + vr > 16:
        col = np.insert(col, vr*2, -1)
    elif self_pos[0] - vr < 0:
        col = np.insert(col, 0, -1)

    # game_state_position["reduced_tiles"] = game_state['field'][max(0,row-vr): min(17,row+vr),max(0,col-vr): min(17,col+vr)]
    game_state_position["self"] = self_pos
    game_state_position["row"] = row
    game_state_position["col"] = col

    return game_state_position

    
def relative_position(items:List[Tuple], agent:Tuple) -> List[Tuple]:
    """
    Takes the original coordinates and recalculates them relative to the agent position as 0,0
    This limits possible states without losing information
    """
    #TODO: Clean this up, the slight alteration item[0] is needed because bomb tuple is ((X,Y),Turns) and Coins just (X,Y)
    try:
        relative_position = [tuple(map(lambda i, j: i - j, item[0], agent)) for item in items]
    except:
        relative_position = [tuple(map(lambda i, j: i - j, item, agent)) for item in items]
    return relative_position

def rotation(self, game_state:Dict):
    """
    #IDEA not used for now

    Rotates all actions as if the agent always starts on the top left.
    This lets every start always look similar and should result in more stable starting strategy
    """
    global ACTIONS

    if game_state["self"][3] == (1,1):
        pass
    if game_state["self"][3] == (15,1):
        pass
    if game_state["self"][3] == (1,15):
        pass
    if game_state["self"][3] == (15,15):
        pass