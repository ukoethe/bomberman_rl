from unittest import TestCase

from agent_code.auto_bomber.transitions import Transitions
import agent_code.auto_bomber.auto_bomber_config as config


class TestTransitions(TestCase):
    def test_monte_carlo_value_estimation(self):
        transitions = Transitions(lambda x: x)
        transitions.add_transition(config.ACTIONS[0], None, None, 17.5)
        transitions.add_transition(config.ACTIONS[1], None, None, 10.)
        transitions.add_transition(config.ACTIONS[2], None, None, 20)
        transitions.add_transition(config.ACTIONS[3], None, None, 40)
        transitions.add_transition(config.ACTIONS[1], None, None, 80)

        numpy_trans = transitions.to_numpy_transitions()

        self.assertEqual(37.5, numpy_trans.monte_carlo_value_estimation(0))
        self.assertEqual(40, numpy_trans.monte_carlo_value_estimation(1))
        self.assertEqual(80, numpy_trans.monte_carlo_value_estimation(4))


