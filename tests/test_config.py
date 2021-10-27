from magnet import config


def test_int():
    assert config.test.foo == 12


def test_float():
    assert config.test.bar == 42.53


def test_string():
    assert config.test.baz == "53"


def test_array():
    assert config.test.qux == [1, 2, 4, 6]


