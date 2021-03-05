import pickle
import random
from collections import namedtuple, deque
from typing import List
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
import events as e
from .callbacks import state_to_features
import numpy as np
from itertools import islice 


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify

TRANSITION_HISTORY_SIZE = 2000 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 16

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

GAMMA = 0.99
LEARNING_RATE = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9999

# Events
SURVIVED_STEP = "SURVIVED_STEP"
DIED_DIRECT_NEXT_TO_BOMB = "DIED_DIRECT_NEXT_TO_BOMB"
ALREADY_KNOW_FIELD = "ALREADY_KNOW_FIELD"
CLOSER_TO_COIN = "CLOSER_TO_COIN"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
   
    self.action_size = len(ACTIONS) #Get size of the action
    self.coordinate_history = deque([], 20)
    
    #Hyperparameters
    self.discount_factor = GAMMA #Disocunt Factor
    self.learning_rate = LEARNING_RATE #Learning Rate

    #Hyperparameters to adjust the Exploitation-Explore tradeoff
    self.epsilon = EXPLORATION_MAX  #Setting the epislon (0= Explore, 1= Exploit)
    self.epsilon_decay = EXPLORATION_DECAY #Adjusting how our epsilon will decay
    self.epsilon_min = EXPLORATION_MIN #Min Epsilon

    self.batch_size = BATCH_SIZE #Batch Size for training the neural network
    self.train_start = TRANSITION_HISTORY_SIZE/2 #If Agent's memory is less, no training is done
    
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # For Training evaluation purposes:
    self.score_in_round = 0
    self.number_game = 0


def append_sample(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.transitions.append(Transition(state_to_features(old_game_state),
                                       self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    
    if old_game_state:
        _, score, bombs_left, (x, y) = old_game_state['self']
        # penalty on loops:
        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            events.append(ALREADY_KNOW_FIELD)

        self.coordinate_history.append((x, y))
        # check whether closer to closest coin
        if new_game_state:
            if (state_to_features(old_game_state) - state_to_features(new_game_state))[5] < 0:
                events.append(CLOSER_TO_COIN)
        
        if 'GOT_KILLED' in events:
            # closer to bomb gives higher penalty:
            bombs = old_game_state['bombs']
            bomb_xys = [xy for (xy, t) in bombs]

            step_distance_bombs = []
            for bomb in bombs:
                x_rel_bomb = bomb[0][0] - x
                y_rel_bomb = bomb[0][1] - y
                step_distance_bombs.append(np.abs(x_rel_bomb) +  np.abs(y_rel_bomb))
            step_distance_closest_bomb = sorted(step_distance_bombs)[0]

            if step_distance_closest_bomb < 2:
                events.append(DIED_DIRECT_NEXT_TO_BOMB)
                
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)
        
    # add transition to remembered transisions:
    append_sample(self, old_game_state, self_action, new_game_state, events)
    self.score_in_round += reward_from_events(self, events)
    #print(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    append_sample(self, last_game_state, last_action, None, events)

    if len(self.transitions) < self.train_start:
        self.number_game += 1
        return
    
    batch = random.sample(self.transitions, self.batch_size)
    X = []
    targets = []

    for state, action, state_next, reward in batch:
        q_update = reward
        
        if state is not None:
            if state_next is not None:
                if self.isFit:
                    q_update = (reward + self.discount_factor * np.amax(self.model.predict(state_next.reshape(1, -1))[0]))
                else:
                    q_update = reward

            if self.isFit:
                q_values = self.model.predict(state.reshape(1, -1))
            else:
                q_values = np.zeros(self.action_size).reshape(1, -1)


            q_values[0][ACTIONS.index(action)] = q_update

            X.append(state)
            targets.append(q_values[0])
    
    self.model.fit(X, targets)
    self.isFit = True
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
        
    with open("my-saved-KNeighborsRegressor-model.pt", "wb") as file:
        pickle.dump(self.model, file)
        
    # For training validation purposes:
    score = np.sum(self.score_in_round)
    game = self.number_game
    
    if game%100 == 0:
        print("game number:", game, "  score:", score, "  memory length:",
                 len(self.transitions), "  epsilon:", self.epsilon)
    
    self.score_in_round = 0
    self.number_game += 1 


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    survive_step = 0.3
    #add reward, if distance to closest coin is decreased 
    game_rewards = {
        # my Events:
        SURVIVED_STEP:  survive_step,
        DIED_DIRECT_NEXT_TO_BOMB: -2*survive_step,
        ALREADY_KNOW_FIELD: -0.1,
        CLOSER_TO_COIN: 0.2,
        
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -survive_step,
        
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 5,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -6* survive_step,   # maybe include later that distance to bomb is included in penatly 

        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: survive_step,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
