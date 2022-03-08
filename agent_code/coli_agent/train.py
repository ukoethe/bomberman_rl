from typing import List

import numpy as np

import events as e

# Custom Events (Ideas)
# TODO: actually implement these
# TODO: set rewards/penalties

# Coins
DECREASED_COIN_DISTANCE = "DECREASED_COIN_DISTANCE"  # move towards nearest coing
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # opposite for balance
# calculation of "coin distance" should take into consideration walls & crates (crates add some distance but don't need to be steered around?)
# penalty for moving towards bomb should be higher than reward for moving towards coin

# Navigation
STAGNATED = "STAGNATED"  # agent is still within 4-tile-radius of location 5 turns ago (4/5 bc of bomb explosion time, idk if it makes sense)
PROGRESSED = "PROGRESSED"  # opposite for balance

# Bombs
FLED = "FLED"  # was in danger zone but didn't get killed when bomb exploded
RETREATED = "REATREATED"  # increased distance towards a bomb in danger zone
SUICIDAL = "SUICIDAL"  # waited or moved towards bomb in danger zone

# Enemies
DECREASED_ENEMY_DISTANCE = "DECREASED_ENEMY_DISTANCE"  # but how do you even reward this? is it good or bad? in what situations which?
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # opposite for balance


def setup_training(self):
    """Sets up training"""
    self.number_of_episodes = 100
    self.exploration_rate = self.exploration_rate_initial


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called once after each time step except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.
    """


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent (?) after the last step of a round."""
    pass


def reward_from_events(self, events: List[str]) -> int:
    """
    Returns a summed up reward/penalty for a given list of events that happened

    Also not assigning reward/penalty to definitely(?) neutral actions MOVE LEFT/RIGHT/UP/DOWN or WAIT.
    """
    # TODO: custom events
    # TODO: different rewards for different learning scenarios?
    game_rewards = {
        e.BOMB_DROPPED: 0.25,  # adjust aggressiveness
        # e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 1,
        e.COIN_FOUND: 0.5,
        # e.CRATE_DESTROYED: 0,  # possibly use if agent isn't destroying enough crates
        e.GOT_KILLED: -5,  # adjust passivity
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,  # you dummy
        # e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 1,  # could possibly lead to not being active
        e.INVALID_ACTION: -1,  # necessary? (maybe for penalizing trying to move through walls/crates)
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def training_loop(self):
    for episode in range(self.number_of_episodes):
        self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end
        ) * np.exp(-self.exploration_decay_rate * episode)
