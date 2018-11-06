import numpy

from helpers import zeros8
from beeclust import BeeClust


def sbt(bees):
    """Sanitize bees types"""
    return [tuple(b) for b in bees]


def test_empty_map_empty_bees():
    b = BeeClust(zeros8((2, 2)))
    assert len(b.bees) == 0


def test_one_bee_in_bees():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = -3
    b = BeeClust(simple_map)
    assert len(b.bees) == 1
    assert sbt(b.bees)[0] == (1, 0)


def test_two_bees_in_bees():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = -5
    simple_map[1, 1] = 4
    b = BeeClust(simple_map)
    assert len(b.bees) == 2
    assert (1, 0) in sbt(b.bees)
    assert (1, 1) in sbt(b.bees)


def test_all_bees_in_bees():
    simple_map = numpy.array(
        [
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [-5, -6, -7, -8],
        ], dtype=numpy.int8)

    b = BeeClust(simple_map)
    assert len(b.bees) == 12
    for x in range(3):
        for y in range(4):
            assert (x, y) in sbt(b.bees)


def test_bees_change_after_tick():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = 1
    b = BeeClust(simple_map, p_changedir=0)
    assert len(b.bees) == 1
    assert sbt(b.bees)[0] == (1, 0)
    b.tick()
    assert len(b.bees) == 1
    assert sbt(b.bees)[0] != (1, 0)
