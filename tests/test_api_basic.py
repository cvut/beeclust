from collections import abc
from helpers import zeros8
from beeclust import BeeClust


def test_is_class():
    assert isinstance(BeeClust, type)


def test_takes_and_exposes_numpy_map():
    simple_map = zeros8((4, 6))
    simple_map[1, 3] = 5
    simple_map[1, 4] = -1
    b = BeeClust(simple_map)
    assert (b.map == simple_map).all()


def test_heatmap_shape():
    simple_map = zeros8((16, 32))
    b = BeeClust(simple_map)
    assert b.heatmap.shape == simple_map.shape


def test_bees_is_a_collection():
    b = BeeClust(zeros8((2, 2)))
    # Collection: anything that is iterable, len()-able and in-able
    # Don't try to subclass it, if you return a list, this check will be OK
    assert isinstance(b.bees, abc.Collection)


def test_swarms_is_a_collection():
    b = BeeClust(zeros8((2, 2)))
    assert isinstance(b.swarms, abc.Collection)


def test_score_is_a_float():
    simple_map = zeros8((4, 6))
    simple_map[1, 3] = 1
    b = BeeClust(simple_map)
    assert isinstance(b.score, float)


def test_tick_returns_int():
    b = BeeClust(zeros8((4, 6)))
    assert isinstance(b.tick(), int)


def test_forget_is_callable():
    b = BeeClust(zeros8((2, 2)))
    b.forget()


def test_recalculate_heat_is_callable():
    b = BeeClust(zeros8((2, 2)))
    b.recalculate_heat()
