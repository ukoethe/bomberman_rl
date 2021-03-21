import pickle
from collections import namedtuple, deque
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
import os

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # , 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_rule/my-saved-model_rule.pt')

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.95

FEATURE_HISTORY_SIZE = 10000  # number of features to use for training
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

        index = ACTION_INDEX[self_action]
        if not self.model_initialised:
            x = old_features
            y = reward_from_events(self, events)
        else:
            x = old_features
            y = reward_from_events(self, events) + GAMMA * self.model[index].predict([new_features.ravel()])[0]
            # print(self_action, x, reward_from_events(self, events), self.model[index].predict([new_features.ravel()])[0])

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
        if not self.model_initialised:
            x = state_to_features(last_game_state)
            y = reward_from_events(self, events)  # initial guess: Q = 0
        else:
            # SARSA
            x = state_to_features(last_game_state)
            y = reward_from_events(self, events) + GAMMA * \
                self.model[index].predict([state_to_features(last_game_state).ravel()])[0]

        self.x[index].append(x)
        self.y[index].append(y)

    if all([(len(x) == FEATURE_HISTORY_SIZE) for x in self.x]):
        print("Fitting model")
        for i, action in enumerate(ACTIONS):
            self.model[i].fit(self.x[i], self.y[i])
            self.x[i].clear()
            self.y[i].clear()
        self.model_initialised = True

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
        e.COIN_COLLECTED: 3,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -3,
        # e.WAITED: -1,
        e.DECREASED_DISTANCE: 1,
        e.INCREASED_DISTANCE: -0.5,  # to avoid loops?
        # e.GOT_KILLED: -5,
        # e.KILLED_SELF: -5
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
