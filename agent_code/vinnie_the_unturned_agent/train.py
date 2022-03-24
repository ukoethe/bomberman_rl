from collections import namedtuple, deque
from random import sample
import numpy as np
from typing import List
import agent_code.rule_based_agent.callbacks as rb_agent
import events as e
from .model import Q_Table
from .utils import state_to_features, ACTIONS
import random
import dill as pickle

# from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 30  # keep only ... last transitions
LAST_POSITION_HISTORY_SIZE = 2
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
REPETITION_EVENT = "REPETITION"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.lastPositions = deque(maxlen=LAST_POSITION_HISTORY_SIZE)

    # save-frequence , not used yet just saving at the end of each round
    self.saves = ...

    # The 'model' in whatever form (NN, QT, MCT ...)
    if self.continue_train:
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.model = Q_Table(self, ACTIONS)
        with open("model.pt", "wb") as file:
            pickle.dump(self.model, file)

    self.batch_size = 10
    # rb_setup(self)
    rb_agent.setup(self)


def train_act(self, game_state: dict):

    if random.uniform(0, 1) > self.model.epsilon:
        # self.action is the unique action chosen by the agent
        action = rb_agent.act(self, game_state)
    else:
        features = state_to_features(game_state)
        action = self.model.choose_action(features)

    self.logger.debug(f"Action taken:{action}")
    return action


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
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
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    # Idea: Add your own events to hand out rewards
    if new_game_state["self"][3] in self.lastPositions:
        events.append(REPETITION_EVENT)

    self.lastPositions.append(new_game_state["self"][3])

    self.transitions.append(
        Transition(
            state_to_features(old_game_state),
            np.where(self.model.actions == self_action)[0],
            state_to_features(new_game_state),
            reward_from_events(self, events),
        )
    )

    if len(self.transitions) > self.batch_size:
        batch = sample(self.transitions, self.batch_size)
        self.model.update_q(batch)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            np.where(self.model.actions == last_action)[0],
            None,
            reward_from_events(self, events),
        )
    )

    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 20,
        e.WAITED: -1,
        e.INVALID_ACTION: -5,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        REPETITION_EVENT: -5,
        e.BOMB_DROPPED: 5.1,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 7,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -10,
        e.SURVIVED_ROUND: 20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
