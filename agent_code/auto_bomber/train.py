import numpy as np
from collections import namedtuple, defaultdict
from typing import List
from tensorboardX import SummaryWriter

import events as e
from agent_code.auto_bomber.callbacks import state_to_features

# This is only an example!
from agent_code.auto_bomber.transitions import Transitions


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    self.transitions = Transitions(state_to_features)
    self.writer = SummaryWriter()
    self.action_q_value = None
    self.loss = []

def game_events_occurred(self, old_game_state: dict, last_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.
    -- > we will collect the transition only here

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param last_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    self.transitions.add_transition(old_game_state, last_action, new_game_state, reward)
    if self.action_q_value:
        self.loss.append((self.action_q_value - reward)**2)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: last entered game state (terminal state?)
    :param last_action: action executed last by agent
    :param events: events occurred before end of round (q: all events or all since last game_events_occurred(..) call?)
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    self.transitions.add_transition(last_game_state, last_action, None, reward)

    numpy_transitions = self.transitions.to_numpy_transitions()
    mean_reward = np.mean(numpy_transitions.rewards)
    self.writer.add_scalar('rewards', mean_reward, last_game_state['round'])

    if self.action_q_value:
        self.loss.append((self.action_q_value - reward)**2)

    mean_loss = np.mean(self.loss)
    self.writer.add_scalar('loss', mean_loss, last_game_state['round'])

    self.model.fit_model_with_transition_batch(self.transitions)
    self.model.store()
    # clear experience buffer for next round
    self.transitions.clear()

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # todo reward definition (right now only first sketch):
    # q: how to determine the winner?
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 75,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -75,
        e.SURVIVED_ROUND: 1000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
