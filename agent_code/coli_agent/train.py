from collections import deque, namedtuple
from typing import List

import numpy as np

import events as e
from agent_code.coli_agent.callbacks import ACTIONS, state_to_features

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions

# --- Custom Events ---
# TODO: actually implement these

# - Events with direct state feature correspondence -

WAS_BLOCKED = (
    "WAS_BLOCKED"  # tried to move into a wall/crate/enemy/explosion (strong penalty)
)
MOVED = "MOVED"  # moved somewhere and wasn't blocked (small reward)

PROGRESSED = "PROGRESSED"  # in last 5 turns, agent visited at least 3 unique tiles

FLED = "FLED"  # was in "danger zone" of a bomb and moved out of it (reward)
SUICIDAL = "SUICIDAL"  # moved from safe field into "danger" zone of bomb (penalty, higher than reward)

# USED_SHORTEST_COIN_PATH = "USED_SHORTEST_COIN_PATH"  # moved along the shortest distance path to nearest coin (used the feature)
# coin distance is calculated using graph pathfinding, takes into consideration walls, crates, explosions + whether another agent is nearer
# penalty for moving towards bomb should be higher than reward for moving towards coin
# if there are no more collectible coins, the coin feature automatically switches to indicating the nearest crate:
# USED_SHORTEST_CRATE_PATH = "USED_SHORTEST_CRATE_PATH"

DECREASED_COIN_DISTANCE = "DECREASED_COIN_DISTANCE"  # decreased length of shortest path to nearest coin BY ONE
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # increased length of shortest path to nearest coin BY ONE
DECREASED_CRATE_DISTANCE = "DECREASED_CRATE_DISTANCE"  # decreased length of shortest path to nearest crate BY ONE
INCREASED_CRATE_DISTANCE = "INCREASED_CRATE_DISTANCE"  # increased length of shortest path to nearest crate BY ONE

INCREASED_SURROUNDING_CRATES = "INCREASED_SURROUNDING_CRATES"  # low reward
DECREASED_SURROUNDING_CRATES = (
    "DECREASED_SURROUNDING_CRATES"  # equal or slightly higher penalty for balance
)

# - Events without direct state feature correspondence -

# idea: Agent-Coin ratio: reward going after crates when there's many coins left (max: 9) and reward going after agents when there aren't
# idea: reward caging enemies

EXLPORE = "EXLPORE"  # slightly reward moving into a new quadrant

DECREASED_NEIGHBORING_WALLS = "DECREASED_NEIGHBORING_WALLS"  # low reward
INCREASED_NEIGHBORING_WALLS = (
    "INCREASED_NEIGHBORING_WALLS"  # low, penalty, penalty higher than reward
)

# more fine-grained bomb area movements
INCREASED_BOMB_DISTANCE = "INCREASED_BOMB_DISTANCE"
DECREASED_BOMB_DISTANCE = "DECREASED_BOMB_DISTANCE"


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
            state_to_features(self, old_game_state, self.history),
            self_action,
            state_to_features(self, new_game_state, self.history),
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

    # if we want to have a wall counting reward, use this to add the event
    # wall_counter = 0
    # neighboring_coordinates = _get_neighboring_tiles(own_position, 1)
    # for coord in neighboring_coordinates:
    #     try:
    #         if game_state["field"][coord] == -1:  # geht das? wer weiß
    #             wall_counter += 1
    #     except IndexError:
    #         print(
    #             "tried to access tile out of bounds (walls)"
    #         )  # TODO remove, just for "debugging"


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent after the last step of a round."""
    self.transitions.append(
        Transition(
            state_to_features(self, last_game_state, self.history),
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

    Not assigning reward/penalty to definitely(?) neutral actions MOVE LEFT/RIGHT/UP/DOWN or WAIT.
    """

    game_rewards = {
        e.BOMB_DROPPED: 5,  # adjust aggressiveness
        # e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 50,
        # e.COIN_FOUND: 5,  # direct consequence from crate destroyed, redundant reward?
        e.WAITED: -3,  # adjust passivity
        e.CRATE_DESTROYED: 4,
        e.GOT_KILLED: -50,  # adjust passivity
        e.KILLED_OPPONENT: 200,
        e.KILLED_SELF: -10,  # you dummy --- this *also* triggers GOT_KILLED
        e.OPPONENT_ELIMINATED: 0.05,  # good because less danger or bad because other agent scored points?
        # e.SURVIVED_ROUND: 0,  # could possibly lead to not being active - actually penalize if agent too passive?
        e.INVALID_ACTION: -1,  # necessary? (maybe for penalizing trying to move through walls/crates) - yes, seems to be necessary to learn that one cannot place a bomb after another placed bomb is still not exploded
        WAS_BLOCKED: -20,
        MOVED: 0.5,
        PROGRESSED: 2,  # higher?
        FLED: 15,
        SUICIDAL: -15,
        DECREASED_COIN_DISTANCE: 8,
        INCREASED_COIN_DISTANCE: -8,  # higher? lower? idk
        DECREASED_CRATE_DISTANCE: 1,
        INCREASED_CRATE_DISTANCE: -1,
        INCREASED_SURROUNDING_CRATES: 1.5,
        # EXLPORE: 2,
        DECREASED_NEIGHBORING_WALLS: 1,
        INCREASED_NEIGHBORING_WALLS: -1,
        # INCREASED_BOMB_DISTANCE: 5,
        # DECREASED_BOMB_DISTANCE: -5
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
