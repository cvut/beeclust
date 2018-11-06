import math
import numpy

from helpers import full8, zeros8
from beeclust import BeeClust


# Expected defaults
T_ENV = 22
T_HEATER = 40
T_COOLER = 5

# Codes
WALL = 5
HEATER = 6
COOLER = 7


def test_heatmap_empty():
    simple_map = zeros8((3, 4))
    b = BeeClust(simple_map)
    assert numpy.isclose(b.heatmap, T_ENV).all()


def test_heatmap_all_heaters():
    simple_map = full8((8, 2), HEATER)
    b = BeeClust(simple_map)
    assert numpy.isclose(b.heatmap, T_HEATER).all()


def test_heatmap_all_coolers():
    simple_map = full8((8, 2), COOLER)
    b = BeeClust(simple_map)
    assert numpy.isclose(b.heatmap, T_COOLER).all()


def test_recalculate_heat():
    simple_map = zeros8((3, 4))
    b = BeeClust(simple_map)
    b.heatmap[:, :] = 100
    assert numpy.isclose(b.heatmap, 100).all()
    b.recalculate_heat()
    assert numpy.isclose(b.heatmap, T_ENV).all()
    b.map[:, :] = HEATER
    b.recalculate_heat()
    assert numpy.isclose(b.heatmap, T_HEATER).all()


def test_heatmap_central_heater():
    simple_map = zeros8((3, 3))
    simple_map[1, 1] = HEATER
    b = BeeClust(simple_map)
    assert b.heatmap[1, 1] == T_HEATER
    for x in range(3):
        for y in range(3):
            if (x, y) != (1, 1):
                assert T_ENV < b.heatmap[x, y] < T_HEATER
                assert math.isclose(b.heatmap[x, y], 38.2)


def test_heatmap_central_cooler():
    simple_map = zeros8((3, 3))
    simple_map[1, 1] = COOLER
    b = BeeClust(simple_map)
    assert b.heatmap[1, 1] == T_COOLER
    for x in range(3):
        for y in range(3):
            if (x, y) != (1, 1):
                assert T_ENV > b.heatmap[x, y] > T_COOLER
                assert math.isclose(b.heatmap[x, y], 6.7)


def test_wall_stops_heat():
    simple_map = zeros8((3, 3))
    simple_map[:, 1] = WALL
    simple_map[:, 2] = HEATER
    b = BeeClust(simple_map)
    assert math.isclose(b.heatmap[0, 0], T_ENV)
    assert math.isclose(b.heatmap[1, 0], T_ENV)
    assert math.isclose(b.heatmap[2, 0], T_ENV)


def test_wall_stops_cool():
    simple_map = zeros8((3, 3))
    simple_map[:, 1] = WALL
    simple_map[:, 2] = COOLER
    b = BeeClust(simple_map)
    assert math.isclose(b.heatmap[0, 0], T_ENV)
    assert math.isclose(b.heatmap[1, 0], T_ENV)
    assert math.isclose(b.heatmap[2, 0], T_ENV)


def test_pseoudorandom_heat():
    simple_map = numpy.array([
        [HEATER, WALL, COOLER],
        [-1, 0, 4],
        [HEATER, COOLER, 0],
        [0, 3, 0],
    ])
    b = BeeClust(simple_map)

    WARM = 22.9
    COLD = 14.8

    assert math.isclose(b.heatmap[0, 0], T_HEATER)
    assert math.isclose(b.heatmap[0, 2], T_COOLER)

    assert math.isclose(b.heatmap[1, 0], WARM)
    assert math.isclose(b.heatmap[1, 1], WARM)
    assert math.isclose(b.heatmap[1, 2], COLD)

    assert math.isclose(b.heatmap[2, 0], T_HEATER)
    assert math.isclose(b.heatmap[2, 1], T_COOLER)
    assert math.isclose(b.heatmap[2, 2], COLD)

    assert math.isclose(b.heatmap[3, 0], WARM)
    assert math.isclose(b.heatmap[3, 1], WARM)
    assert math.isclose(b.heatmap[3, 2], COLD)


def test_heat_distribution_along_diagonal():
    simple_map = zeros8((8, 8))
    simple_map[0, 0] = HEATER
    b = BeeClust(simple_map)
    TEMPS = [40, 38.2, 30.1, 27.4, 26.05, 25.24, 24.7, 24.31428571]
    for i in range(-8, 8):
        assert numpy.isclose(b.heatmap.diagonal(i), TEMPS[abs(i):]).all()


def test_custom_temperatures():
    simple_map = zeros8((4, 4))
    simple_map[0, -1] = HEATER
    simple_map[-1, 0] = COOLER
    cooler, env, heater = -20, 0, 20
    b = BeeClust(simple_map, T_cooler=cooler, T_env=env, T_heater=heater)
    assert math.isclose(b.heatmap[0, -1], heater)
    assert math.isclose(b.heatmap[-1, 0], cooler)
    assert numpy.isclose(b.heatmap.diagonal(), env).all()
    assert math.isclose(b.heatmap[1, -2], 9)
    assert math.isclose(b.heatmap[-2, 1], -9)


def test_custom_temperatures_custom_k():
    simple_map = zeros8((4, 4))
    simple_map[0, -1] = HEATER
    simple_map[-1, 0] = COOLER
    cooler, env, heater = -20, 0, 20
    b = BeeClust(simple_map, T_cooler=cooler, T_env=env,
                 T_heater=heater, k_temp=.8)
    assert math.isclose(b.heatmap[0, -1], heater)
    assert math.isclose(b.heatmap[-1, 0], cooler)
    assert numpy.isclose(b.heatmap.diagonal(), env).all()
    assert math.isclose(b.heatmap[1, -2], 8)
    assert math.isclose(b.heatmap[-2, 1], -8)


def test_score_one_bee():
    simple_map = zeros8((3, 4))
    simple_map[1, 1] = 3
    b = BeeClust(simple_map)
    assert math.isclose(b.score, T_ENV)


def test_score_one_bee_heater():
    simple_map = zeros8((3, 4))
    simple_map[1, 1] = -2
    simple_map[1, 2] = HEATER
    b = BeeClust(simple_map)
    assert T_HEATER > b.score > T_ENV
    assert math.isclose(b.score, 38.2)


def test_score_two_bees_heater_cooler():
    simple_map = zeros8((3, 4))
    simple_map[0, 0] = COOLER
    simple_map[2, 3] = HEATER
    simple_map[0, 1] = 2
    simple_map[2, 2] = 3
    b = BeeClust(simple_map)
    assert math.isclose(b.score, 22.675)


def test_score_changes():
    simple_map = zeros8((3, 1))
    simple_map[1, 0] = 1
    simple_map[2, 0] = HEATER
    b = BeeClust(simple_map, p_changedir=0)
    score = b.score
    b.tick()
    assert b.score < score
