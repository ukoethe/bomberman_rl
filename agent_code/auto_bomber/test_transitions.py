from unittest import TestCase

import numpy as np

from agent_code.auto_bomber.transitions import Transitions
import agent_code.auto_bomber.auto_bomber_config as config

ARR_SIZE = 10


class TestTransitions(TestCase):
    def test_monte_carlo_value_estimation(self):
        transitions = Transitions(lambda x: x)
        transitions.add_transition(None, config.ACTIONS[0], None, 17.5)
        transitions.add_transition(None, config.ACTIONS[1], None, 10.)
        transitions.add_transition(None, config.ACTIONS[2], None, 20)
        transitions.add_transition(None, config.ACTIONS[3], None, 40)
        transitions.add_transition(None, config.ACTIONS[1], None, 80)

        numpy_trans = transitions.to_numpy_transitions()

        self.assertEqual(37.5, numpy_trans.monte_carlo_value_estimation(0))
        self.assertEqual(40, numpy_trans.monte_carlo_value_estimation(1))
        self.assertEqual(80, numpy_trans.monte_carlo_value_estimation(4))

    def test_get_features_and_value_estimates(self):
        transitions = Transitions(lambda x: x)

        transitions.add_transition(np.zeros((ARR_SIZE,)), config.ACTIONS[0], np.ones((ARR_SIZE,)), 10)
        transitions.add_transition(np.ones(ARR_SIZE), config.ACTIONS[1], np.full((ARR_SIZE,), 2), 20.)
        transitions.add_transition(np.full((ARR_SIZE,), 2), config.ACTIONS[0], np.full((ARR_SIZE,), 3), 40)
        transitions.add_transition(np.full((ARR_SIZE,), 3), config.ACTIONS[1], np.full((ARR_SIZE,), 4), 80)
        transitions.add_transition(np.full((ARR_SIZE,), 4), config.ACTIONS[0], np.full((ARR_SIZE,), 5), 160)
        transitions.add_transition(np.full((ARR_SIZE,), 5), config.ACTIONS[1], np.full((ARR_SIZE,), 6), 320)
        transitions.add_transition(np.full((ARR_SIZE,), 6), config.ACTIONS[0], np.full((ARR_SIZE,), 7), 640)

        numpy_trans = transitions.to_numpy_transitions()
        x_0_all, y_0_all = numpy_trans.get_features_and_value_estimates(config.ACTIONS[0])
        self.assertEqual((4, 10), x_0_all.shape)
        self.assertEqual((4,), y_0_all.shape)

        exp_x_0 = np.array([np.zeros((ARR_SIZE,)), np.full((ARR_SIZE,), 2), np.full((ARR_SIZE,), 4), np.full((ARR_SIZE,), 6)])
        np.testing.assert_array_equal(x_0_all, exp_x_0)
        exp_y_0 = np.array([70, 200, 480, 640])
        np.testing.assert_array_equal(y_0_all, exp_y_0)

        x_1_all, y_1_all = numpy_trans.get_features_and_value_estimates(config.ACTIONS[1])
        self.assertEqual((3, 10), x_1_all.shape)
        self.assertEqual((3,), y_1_all.shape)

        exp_x_1 = np.array([np.ones((ARR_SIZE,)), np.full((ARR_SIZE,), 3), np.full((ARR_SIZE,), 5)])
        np.testing.assert_array_equal(x_1_all, exp_x_1)
        exp_y_1 = np.array([120, 320, 640])
        np.testing.assert_array_equal(y_1_all, exp_y_1)

    def test_get_features_and_value_estimates_single_action(self):
        transitions = Transitions(lambda x: x)

        transitions.add_transition(np.zeros((ARR_SIZE,)), config.ACTIONS[0], np.ones((ARR_SIZE,)), 10)
        numpy_trans = transitions.to_numpy_transitions()

        x_0_all, y_0_all = numpy_trans.get_features_and_value_estimates(config.ACTIONS[0])
        self.assertEqual((1, 10), x_0_all.shape)
        self.assertEqual((1,), y_0_all.shape)






