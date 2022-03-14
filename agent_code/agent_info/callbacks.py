from collections import defaultdict
import os
import pickle
import random
from typing import Dict, List, Tuple
from unicodedata import category
import agent_code.agent_info.train as train
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
    if self.train or not os.path.isfile("q-table.npy"):

        self.logger.info("Training model.")

        # Deactivate if u want to train a completly new agent
        if os.path.isfile("q-table.npy") and True:
            self.continue_train = True
            
        self.action_space_size = len(ACTIONS)
        train.setup_training(self)
        
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
    #TODO: Exploration vs exploitation
    random_prob = .1

    #TODO: Remove this here and dynamicly handle this while training
    self.explore = 0.05
    if self.train and self.explore < random_prob:

        self.logger.debug("Choosing action based on training algorith.")
        return train.act(self, game_state)

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
    # Add/Adjust game_state in q_table

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
    
    features = defaultdict

    walls, creates = awarness(game_state)

    return features

    
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

def rotation(game_state:Dict):

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

def vision_field(game_state:Dict):

    # Position of the agent
    self_pos = game_state["self"][3]

    # How far can you look
    vision = 1

    # Game Field at the position of the agent
    left  = self_pos[0] - vision
    right = self_pos[0] + vision + 1

    top  = self_pos[0] - vision
    down = self_pos[0] + vision + 1

    return game_state["field"][left:right, down:top]

def awarness(self, game_state:Dict):
    """
    With this their is no need to pass the 
    """
    vis_field = vision_field(game_state)

    walls = np.where(vis_field == -1)
    crates = np.where(vis_field == 1)

    # Brings situational awerness feature vectors
    walls = np.array([walls[0, 1], walls[1, 0], walls[2, 1], walls[1, 2]])
    crates = np.array([crates[0, 1], crates[1, 0], crates[2, 1], crates[1, 2]])

    return walls, crates

def danger(game_state:Dict):

    # implement danger posed by bombs that are about to set off
    ...
    
