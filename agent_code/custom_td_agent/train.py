import pickle
from collections import namedtuple, deque
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
import os


import numpy as np
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT' , 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_rule/my-saved-model_rule.pt')

from sklearn.utils.validation import check_is_fitted

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.95
POSITION_HISTORY_SIZE = 5

FEATURE_HISTORY_SIZE = 500  # number of features to use for training
# train with:
# python main.py play --agents rule_based_agent --no-gui --train 1 --n-rounds 1200
# with random_prob = 0.7 in rule_based_agent/callbacks.py

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model_not_fitted = [False] * len(ACTIONS)
    self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)
    self.x = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # features
    self.y = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # targets
    if not self.model:  # initialise model
        print("initialising model")
        self.model = [
            GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_depth=1,  # default was learning_rate=0.1
                                      random_state=0, loss='ls', warm_start=True, init='zero') for _ in ACTIONS]
        self.model_initialised = False
    else:
        self.model_initialised = True  # TODO check if model is fitted
        for index, model in enumerate(self.model):
            try:
                model.predict([[0,0,0,0,0,0,0,0,0,0]])
            except NotFittedError:
                self.model_not_fitted[index] = True


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

    if old_game_state is not None and new_game_state is not None and self_action is not None:  # TODO discard if states are same?
        # state_to_features is defined in callbacks.py
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)
        if new_features[0]**2 + new_features[1]**2 < old_features[0]**2 + old_features[1]**2:
            events.append(e.DECREASED_DISTANCE)
        if new_features[0]**2 + new_features[1]**2 > old_features[0]**2 + old_features[1]**2:
            events.append(e.INCREASED_DISTANCE)

        # check for loop and add reward if there is one
        current_position = new_game_state['self'][3]
        if len(self.position_history) > 4 and current_position == self.position_history[1] and current_position == self.position_history[3]:
            events.append(e.STUCK_IN_LOOP)
            #print('appended loop reward')

        self.position_history.append(current_position)


        index = ACTION_INDEX[self_action]
        if not self.model_initialised or True in self.model_not_fitted:
            x = old_features
            y = reward_from_events(self, events)
        else:
            old_features = state_to_features(old_game_state)
            new_features = state_to_features(new_game_state)
            x = old_features
            x_new = new_features
            y = reward_from_events(self, events) + GAMMA * np.max(np.ravel([model.predict([x_new.ravel()]) for model in self.model]))


        self.x[index].append(x)
        self.y[index].append(y)


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

    if last_action is not None:
        index = ACTION_INDEX[last_action]
        if not self.model_initialised or True in self.model_not_fitted:
            x = state_to_features(last_game_state)
            y = reward_from_events(self, events)  # initial guess: Q = 0
        else:
            # SARSA
            old_features = state_to_features(last_game_state)
            x = old_features
            y = reward_from_events(self, events)

        self.x[index].append(x)
        self.y[index].append(y)
    
    """ for i, x in enumerate(self.x):
        print(len(x))
        if len(x) == FEATURE_HISTORY_SIZE:
            print("Fitting model")
            self.model[i].fit(self.x[i], self.y[i])
            self.x[i].clear()
            self.y[i].clear()
            self.model_not_fitted[i] = False
        self.model_initialised = True """

    with open('steps.txt', 'a') as steps_log:
        steps_log.write(str(last_game_state['step']) + "\t")

    with open('scores.txt', 'a') as scores_log:
        scores_log.write(str(last_game_state['self'][1]) + "\t")

    # Store the model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -3,
        e.DECREASED_DISTANCE: 1,
        e.INCREASED_DISTANCE: -0.5,  # to avoid loops?
        e.WAITED: -0.5,
        e.BOMB_DROPPED: 0.1,
        e.CRATE_DESTROYED: 5,
        # e.GOT_KILLED: -5,
        e.KILLED_SELF: -1,
        e.MOVED_DOWN: -0.5,
        e.MOVED_UP: -0.5,
        e.MOVED_LEFT: -0.5,
        e.MOVED_RIGHT: -0.5,
        e.STUCK_IN_LOOP: -5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event == 'STUCK_IN_LOOP':
            self.logger.info(f'agent stuck in loop -> handed out punishment')
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
