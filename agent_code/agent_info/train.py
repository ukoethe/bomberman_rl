from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List
from agent_code.rule_based_agent.callbacks import act as rb_act, setup as rb_setup
import events as e
# from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

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

    # How a game-state looks to adjust the model input accordingly
    self.format = ...

    # The 'model' in whatever form (NN, QT, MCT ...) 
    self.model = q_table(self)

    rb_setup(self)


def act(self, gamestate):
    return rb_act(self, gamestate)

def q_table(self):

    # Enables us to continue training on an already started Q-Table
    # This is useful when base training it with replays from other agents and then just tuning it with the new one
    # or when just want to continue training the same agent with more episodes
    self.continue_train = False
    if not self.continue_train:
        state_space_size = 10_000

        self.q_table = np.zeros([state_space_size, self.action_space_size])
    
    else:
        self.q_table = np.load("q_table.npy")

        assert np.size(self.q_table)[1] == self.action_space_size, "To continue training an old model you need to have the same action space"

def q_params(self):
    # Medium implementing an iterable q-table
    num_episodes = ...
    max_steps_ep = ...

    lr = ...
    discount = ...

    exploration = ...
    max_exploration = ...
    min_exploration = ...
    exploration_decay = ...

    """
    # How a game works
    for episode in range(num_episodes):
        for step in range(max_steps):
            # Exploit/Explore Trade-off
            # Take new action
            # Update Q
            # set new state
            # add new reward
        
        # Exploit Rate Decay
        # Add reward to total reward list
    """
    


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
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
