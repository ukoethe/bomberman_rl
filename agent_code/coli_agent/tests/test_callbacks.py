from coli_agent.callbacks import DecisionTransformer

def test_adding_one():
    dt = DecisionTransformer()
    result = dt.adding_one(5)
    assert result == 6