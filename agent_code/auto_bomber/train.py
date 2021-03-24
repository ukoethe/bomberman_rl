from queue import Queue
from typing import List

from agent_code.auto_bomber import custom_events as ce
from agent_code.auto_bomber.feature_engineering import state_to_features
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

    self.q = Queue(maxsize=self.model.hyper_parameters["region_time_tolerance"])


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
    # state_to_features is defined in callbacks.py
    self.transitions.add_transition(old_game_state, last_action, new_game_state, reward_from_events(self, events))
    # Punishment, if agent is still in the same radius after certain time steps
    new_position = new_game_state["self"][3]
    region_size = self.model.hyper_parameters["region_size"]
    if self.q.full():
        old_position = self.q.get()
        if (old_position[0] - region_size <= new_position[0] <= old_position[0] + region_size) \
                or (old_position[1] - region_size <= new_position[1] <= old_position[1] + region_size):
            events.append(ce.SAME_REGION)
    self.q.put(new_position)


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
    self.transitions.add_transition(last_game_state, last_action, None, reward_from_events(self, events))

    self.model.fit_model_with_transition_batch(self.transitions, last_game_state['round'])
    self.model.store()
    # clear experience buffer for next round
    self.transitions.clear()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # q: how to determine the winner?

    rewards_dict = self.model.hyper_parameters["game_rewards"]
    reward_sum = 0
    for event in events:
        if event in rewards_dict:
            reward_sum += rewards_dict[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
