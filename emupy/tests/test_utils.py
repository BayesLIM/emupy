from emupy import utils

def test_flatten():
    a = [[1, 2, 3, 4, 5]]
    aflat = utils.flatten(a)
    assert aflat == [1, 2, 3, 4, 5]