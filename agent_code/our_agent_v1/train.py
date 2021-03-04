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


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify

TRANSITION_HISTORY_SIZE = 2000 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 16

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT','BOMB']
GAMMA = 0.99
LEARNING_RATE = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9999

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
   
    self.action_size = len(ACTIONS) #Get size of the action

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
    if not events:
        events.append(PLACEHOLDER_EVENT)

    # add transition to remembered transisions:
    append_sample(self, old_game_state, self_action, new_game_state, events)
    self.score_in_round += reward_from_events(self, events)


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
    game_rewards = {
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP:-0.1,
        e.MOVED_DOWN: -0.1,
        e.WAITED: -0.1,
        e.INVALID_ACTION: -5,

        e.BOMB_DROPPED: -5,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 500,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -500,

        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 5,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
