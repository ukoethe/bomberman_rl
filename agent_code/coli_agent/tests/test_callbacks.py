from coli_agent.callbacks import DecisionTransformer

# game_state_beginning = {
#     # TODO
# }


def test_adding_one():
    decision_transformer = DecisionTransformer()
    result = decision_transformer.adding_one(5)
    assert result == 6


# def test_wall_features(game_state_beginning):
#     "Test if wall feature correctly counts walls"
#     f = state_to_features(self, game_state_beginning)
#     assert f[0] == 2
