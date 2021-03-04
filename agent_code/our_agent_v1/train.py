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
TRANSITION_HISTORY_SIZE = 30 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT','BOMB']
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 15

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.999

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.exploration_rate = EXPLORATION_MAX
    self.action_space =  len(ACTIONS)
    #self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
    self.model = KNeighborsRegressor(n_jobs=-1)
    #self.model = MultiOutputRegressor(SVR(), n_jobs=8)
    self.isFit = False


def remember(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.transitions.append(Transition(state_to_features(old_game_state),
                                       self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    
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
    remember(self, old_game_state, self_action, new_game_state, events)


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
    
    remember(self, last_game_state, last_action, None, events)
    #print('In the last game the follwing actions where executed:' , events)
    #print('The reward of the last game was:' , reward_from_events(self, events))

    if len(self.transitions) < BATCH_SIZE:
        return
    
    batch = random.sample(self.transitions, BATCH_SIZE)
    X = []
    targets = []

    for state, action, state_next, reward in batch:
        q_update = reward
        #print(reward)
        if state is not None:
            if state_next is not None:
                if self.isFit:
                    q_update = (reward + GAMMA * np.amax(self.model.predict(state_next.reshape(1, -1))[0]))
                else:
                    q_update = reward

            if self.isFit:
                q_values = self.model.predict(state.reshape(1, -1))
            else:
                q_values = np.zeros(self.action_space).reshape(1, -1)


            q_values[0][ACTIONS.index(action)] = q_update

            X.append(state)
            targets.append(q_values[0])
    
    self.model.fit(X, targets)
    self.isFit = True
    self.exploration_rate *= EXPLORATION_DECAY
    self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
        e.INVALID_ACTION: -1,

        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 500,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -200,

        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 100,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
