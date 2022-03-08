from collections import deque, namedtuple
from typing import List

import numpy as np

import events as e
from agent_code.coli_agent.callbacks import state_to_features

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions


def setup_training(self):
    """Sets up training"""
    self.exploration_rate = self.exploration_rate_initial
    # (s, a, s', r)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode = 0  # need to keep track of episodes
    self.rewards_of_episode = 0


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called once after each time step except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.
    """
    return


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent after the last step of a round."""
    reward = reward_from_events(self, events)
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward)
    )
    self.rewards_of_episode += reward

    self.logger.info(
        f"Total rewards in episode {self.episode}: {self.rewards_of_episode}"
    )
    self.rewards_of_episode = 0

    np.save(f"q_table-{self.timestamp}", self.q_table)

    self.episode += 1
    self.exploration_rate = self.exploration_rate_end + (
        self.exploration_rate_initial - self.exploration_rate_end
    ) * np.exp(
        -self.exploration_decay_rate * self.episode
    )  # decay


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {e.COIN_COLLECTED: 1, e.KILLED_OPPONENT: 5}
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
