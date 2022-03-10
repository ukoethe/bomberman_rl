from collections import deque, namedtuple
from typing import List

import numpy as np

import events as e
from agent_code.coli_agent.callbacks import ACTIONS, state_to_features

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions


# Custom Events (Ideas)
# TODO: actually implement these
# TODO: set rewards/penalties

# Coins
DECREASED_COIN_DISTANCE = "DECREASED_COIN_DISTANCE"  # move towards nearest coin, penalty should be higher than reward
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # opposite for balance
# calculation of "coin distance" should take into consideration walls & crates (interpret crates as walls)
# take into consideration whether another agent is closer to the coin (nearest coin = coin is nearest to us and we are, out of all agents, nearest to coin)
# penalty for moving towards bomb should be higher than reward for moving towards coin

# Crates
# have nearest crate feature and reward going there -- take into consideration where there are many crates near (e.g. 5 crates two files away is better than 2 crates 1 tile away)
# Agent-Coin ratio: reward going after crates when there's many coins left (max: 9) and reward going after agents when there aren't

# Navigation
# STAGNATED = "STAGNATED"  # agent is still within 4-tile-radius of location 5 turns ago (4/5 bc of bomb explosion time, idk if it makes sense)
# PROGRESSED = "PROGRESSED"  # opposite for balance
# EXLPORE = "EXLPORE" # reward moving away from starting position/quadrant
REVISITED_TILE = "REVISITED_TILE"  # low penalty (bc sometimes necessary to survive)
NEW_TILE = "NEW_TILE"  # low reward, but a bit higher than the penalty

# Walls
DECREASED_NEIGHBORING_WALLS = "DECREASED_NEIGHBORING_WALLS"  # low reward
INCREASED_NEIGHBORING_WALLS = (
    "INCREASED_NEIGHBORING_WALLS"  # low, penalty, penalty higher than reward
)

# Bombs
FLED = "FLED"  # was in danger zone but didn't get killed when bomb exploded
RETREATED = "RETREATED"  # increased distance towards a bomb in danger zone
SUICIDAL = "SUICIDAL"  # waited or moved towards bomb in danger zone, penalty higher than RETREATED reward

# Enemies
# DECREASED_ENEMY_DISTANCE = "DECREASED_ENEMY_DISTANCE"  # but how do you even reward this? is it good or bad? in what situations which?
# INCREASED_ENEMY_DISTANCE = "INCREASED_ENEMY_DISTANCE"  # opposite for balance
# do not include as events, but do include a feature "enemy distance" as weighted sum (distance to nearest is 4x as important as distance to farthest)
# idea: reward caging enemies


def setup_training(self):
    """Sets up training"""
    self.exploration_rate = self.exploration_rate_initial
    self.learning_rate = 0.1
    self.discount_rate = 0.99

    # (s, a, s', r)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode = 0  # need to keep track of episodes
    self.rewards_of_episode = 0


def game_events_occurred(
    self, old_game_state, self_action: str, new_game_state, events
):
    """Called once after each time step except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.

    Will call state_to_features, and can then use these features for adding our custom events.
    (if features = ... -> events.append(OUR_EVENT)). But events can also be added independently of features,
    just using game state in general. Leveraging of features more just to avoid code duplication.
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    # skip first timestep
    if old_game_state is None:
        return

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(
            state_to_features(old_game_state, self.history),
            self_action,
            state_to_features(new_game_state, self.history),
            reward_from_events(self, events),
        )
    )
    state, action, next_state, reward = (
        self.transitions[-1][0],
        self.transitions[-1][1],
        self.transitions[-1][2],
        self.transitions[-1][3],
    )

    action_idx = ACTIONS.index(action)
    self.logger.debug(action_idx)

    self.rewards_of_episode += reward
    self.q_table[state, action_idx] = self.q_table[
        state, action_idx
    ] + self.learning_rate * (
        reward
        + self.discount_rate * np.max(self.q_table[next_state])
        - self.q_table[state, action_idx]
    )
    self.logger.debug(f"Updated q-table: {self.q_table}")


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent after the last step of a round."""
    self.transitions.append(
        Transition(
            state_to_features(last_game_state, self.history),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )
    self.rewards_of_episode += self.transitions[-1][3]

    self.logger.info(
        f"Total rewards in episode {self.episode}: {self.rewards_of_episode}"
    )
    self.rewards_of_episode = 0

    if self.episode % 250 == 0:
        np.save(f"q_table-{self.timestamp}", self.q_table)

    self.episode += 1
    self.exploration_rate = self.exploration_rate_end + (
        self.exploration_rate_initial - self.exploration_rate_end
    ) * np.exp(
        -self.exploration_decay_rate * self.episode
    )  # decay


def reward_from_events(self, events: List[str]) -> int:
    """
    Returns a summed up reward/penalty for a given list of events that happened

    Also not assigning reward/penalty to definitely(?) neutral actions MOVE LEFT/RIGHT/UP/DOWN or WAIT.
    """
    # TODO: custom events
    # TODO: different rewards for different learning scenarios?
    game_rewards = {
        e.BOMB_DROPPED: 2,  # adjust aggressiveness
        # e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 10,
        # e.COIN_FOUND: 5,  # direct consequence from crate destroyed, redundant reward?
        e.WAITED: -1,  # adjust passivity
        e.CRATE_DESTROYED: 4,
        e.GOT_KILLED: -5,  # adjust passivity
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -10,  # you dummy --- this *also* triggers GOT_KILLED
        e.OPPONENT_ELIMINATED: 0.05,  # good because less danger or bad because other agent scored points?
        # e.SURVIVED_ROUND: 0,  # could possibly lead to not being active - actually penalize if agent too passive?
        e.INVALID_ACTION: -1,  # necessary? (maybe for penalizing trying to move through walls/crates) - yes, seems to be necessary to learn that one cannot place a bomb after another placed bomb is still not exploded
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
