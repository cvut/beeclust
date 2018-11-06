import numpy

from helpers import zeros8
from beeclust import BeeClust


def swt(swarms):
    """Sanitize/sort swarms types"""
    return sorted(sorted(tuple(b) for b in s) for s in swarms)


def test_empty_map_empty_swarms():
    b = BeeClust(zeros8((2, 2)))
    assert len(b.swarms) == 0


def test_one_bee_in_swarms():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = -3
    b = BeeClust(simple_map)
    assert swt(b.swarms) == [[(1, 0)]]


def test_two_bees_in_swarms():
    simple_map = zeros8((3, 3))
    simple_map[1, 0] = -5
    simple_map[0, 2] = 4
    b = BeeClust(simple_map)
    assert len(b.swarms) == 2
    assert swt(b.swarms) == [[(0, 2)], [(1, 0)]]


def test_two_bees_in_swarms_corner():
    simple_map = zeros8((3, 3))
    simple_map[1, 0] = -5
    simple_map[0, 1] = 4
    b = BeeClust(simple_map)
    assert len(b.swarms) == 2
    assert swt(b.swarms) == [[(0, 1)], [(1, 0)]]


def test_two_bordering_bees_in_swarms():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = -5
    simple_map[1, 1] = 4
    b = BeeClust(simple_map)
    assert len(b.swarms) == 1
    assert swt(b.swarms) == [[(1, 0), (1, 1)]]


def test_all_bees_in_swarms():
    simple_map = numpy.array(
        [
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [-5, -6, -7, -8],
        ], dtype=numpy.int8)

    b = BeeClust(simple_map)
    assert len(b.swarms) == 1
    assert swt(b.swarms) == [[(x, y) for x in range(3) for y in range(4)]]


def test_swarms_change_after_tick():
    simple_map = zeros8((2, 2))
    simple_map[1, 0] = 1
    b = BeeClust(simple_map, p_changedir=0)
    assert swt(b.swarms) == [[(1, 0)]]
    b.tick()
    assert len(b.swarms) == 1
    assert len(swt(b.swarms)[0]) == 1
    assert swt(b.swarms)[0][0] != (1, 0)
