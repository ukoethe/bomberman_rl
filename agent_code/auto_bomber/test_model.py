from unittest import TestCase
from unittest.mock import patch, mock_open, Mock, MagicMock
import numpy as np
import pickle

import os
from agent_code.auto_bomber.model import LinearAutoBomberModel
from agent_code.auto_bomber.transitions import Transitions, NumpyTransitions
import auto_bomber_config as config


def x_y_for_actions(action):
    if action == config.ACTIONS[0]:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]), np.array([11, 21, 41, 0])
    # else
    return np.array([]), np.array([])


class TestLinearAutoBomberModel(TestCase):

    @patch.object(os.path, attribute="isfile")
    def test_select_best_action(self, mock_path):
        mock_path.isfile.return_value = True

        # get rid of randomness for testing here
        np.random.seed(0)

        read_array = pickle.dumps(
            np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.float32))
        open_mock = mock_open(read_data=read_array)
        with patch('builtins.open', open_mock):
            feature_array = np.array([3, 2, 1], dtype=np.float32)
            elem = LinearAutoBomberModel(lambda x: feature_array)
            action = elem.select_best_action({}, None)

            self.assertEqual("UP", action)

    @patch.object(os.path, attribute="isfile")
    def test_fit_model(self, mock_path):
        mock_path.isfile.return_value = True
        read_array = pickle.dumps(np.ones((6, 3)))
        open_mock = mock_open(read_data=read_array)
        with patch('builtins.open', open_mock):
            transitions = Transitions(None)
            numpy_trans = NumpyTransitions(transitions)
            transitions.to_numpy_transitions = MagicMock(return_value=numpy_trans)
            numpy_trans.get_features_and_value_estimates = MagicMock(side_effect=x_y_for_actions)

            model = LinearAutoBomberModel(None)
            model.fit_model_with_transition_batch(transitions)

            np.testing.assert_array_equal(model.weights[0, :], np.array([1.25, 1.5, 2]))
            np.testing.assert_array_equal(model.weights[1, :], np.ones(3, ))
            np.testing.assert_array_equal(model.weights[2, :], np.ones(3, ))
            np.testing.assert_array_equal(model.weights[3, :], np.ones(3, ))
            np.testing.assert_array_equal(model.weights[4, :], np.ones(3, ))
            np.testing.assert_array_equal(model.weights[5, :], np.ones(3, ))
