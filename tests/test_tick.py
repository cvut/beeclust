import math
import numpy

from helpers import full8, zeros8
from beeclust import BeeClust


def loner(what):
    """Puts {what} into the central field, surrounded by nothing but sorrow"""
    simple_map = zeros8((3, 3))
    simple_map[1, 1] = what
    return simple_map


def test_tick_empty():
    original = zeros8((3, 4))
    original[1, 2] = 5  # wall
    original[1, 3] = 6  # heater
    b = BeeClust(original.copy())
    assert b.map is not original

    # this could go to infinity, but 42 would do
    # no monkey business at 43, please
    for _ in range(42):
        assert b.tick() == 0
        assert (b.map == original).all()


def test_tick_full():
    simple_map = full8((8, 12), 2)
    b = BeeClust(simple_map)
    for _ in range(42):
        assert b.tick() == 0


def test_tick_small():
    simple_map = full8((1, 1), 3)
    b = BeeClust(simple_map)
    for _ in range(42):
        assert b.tick() == 0


def test_bee_goes_up():
    b = BeeClust(loner(1), p_changedir=0)
    assert b.tick() == 1
    assert b.map[0, 1] == 1
    assert b.map.sum() == 1


def test_bee_goes_right():
    b = BeeClust(loner(2), p_changedir=0)
    assert b.tick() == 1
    assert b.map[1, 2] == 2
    assert b.map.sum() == 2


def test_bee_goes_down():
    b = BeeClust(loner(3), p_changedir=0)
    assert b.tick() == 1
    assert b.map[2, 1] == 3
    assert b.map.sum() == 3


def test_bee_goes_left():
    b = BeeClust(loner(4), p_changedir=0)
    assert b.tick() == 1
    assert b.map[1, 0] == 4
    assert b.map.sum() == 4


def test_bee_doesnt_go_up():
    b = BeeClust(loner(1), p_changedir=1)
    assert b.tick() == 1
    assert b.map[0, 1] == 0  # doesn't go up
    assert b.map[1, 1] == 0  # doesn't stay
    assert numpy.count_nonzero(b.map) == 1  # is somewhere
    # bee doesn't go left/right/down are left as an exercise for the reader


def test_bee_waits_and_moves():
    b = BeeClust(loner(-3))
    assert b.tick() == 0
    assert b.map[1, 1] == -2
    assert b.tick() == 0
    assert b.map[1, 1] == -1
    assert b.tick() == 0
    assert 0 < b.map[1, 1] < 5
    assert b.tick() == 1


def test_bee_keeps_going_right_until_it_hits_end():
    b = BeeClust(numpy.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), p_changedir=0)
    for j in range(1, 10):
        assert b.tick() == 1
        assert numpy.count_nonzero(b.map) == 1
        assert b.map[0, j] == 2
    assert b.tick() == 0
    assert b.map[0, -1] != 0


def test_bee_keeps_going_left_until_it_hits_end():
    b = BeeClust(numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 4]]), p_changedir=0)
    for j in range(2, 11):
        print(j)
        assert b.tick() == 1
        assert numpy.count_nonzero(b.map) == 1
        assert b.map[0, -j] == 4
    assert b.tick() == 0
    assert b.map[0, 0] != 0


def test_bee_keeps_going_down_until_it_hits_a_barrier():
    for wall in 5, 6, 7:
        b = BeeClust(numpy.array([[3], [0], [0], [0], [0], [wall], [0], [0]]),
                     p_changedir=0)
        for j in range(1, 5):
            assert b.tick() == 1
            assert numpy.count_nonzero(b.map) == 2
            assert b.map[j, 0] == 3
        assert b.tick() == 0
        assert b.map[4, 0] != 0


def test_bee_is_moving_more_or_less_random():
    # WARNING: This test is not deterministic, if it fails, run it again.
    # For a couple of p_changedir probabilities, it ticks ~thousands times,
    # it counts how many times the bee actually changes the direction.
    # The numbers should be more or less reflect the p_changedir value.
    # Remaining directions should be selected by a fair chance.
    # After some experimenting, we've set tolerance to be 30 %,
    # that should be enough to beat the odds. But well, it's random.
    total = 1024
    for p_changedir in .2, .4, .5, .9:
        ups = downs = lefts = rights = 0
        for _ in range(total):
            b = BeeClust(loner(1), p_changedir=p_changedir)
            assert b.tick() == 1
            assert numpy.count_nonzero(b.map) == 1
            if b.map[0, 1]:
                ups += 1
            elif b.map[2, 1]:
                downs += 1
            elif b.map[1, 0]:
                lefts += 1
            else:
                assert b.map[1, 2]
                rights += 1
        tolreance = .3
        changedir = p_changedir * total
        goes = total - changedir
        assert math.isclose(ups, goes, rel_tol=tolreance)
        assert math.isclose(downs, changedir/3, rel_tol=tolreance)
        assert math.isclose(lefts, changedir/3, rel_tol=tolreance)
        assert math.isclose(rights, changedir/3, rel_tol=tolreance)


def test_hit_wall_turns_and_goes():
    b = BeeClust(numpy.array([[0, 2, 5]]), p_wall=0, p_changedir=0)
    assert b.tick() == 0
    assert (b.map == [[0, 4, 5]]).all()


def test_hit_wall_stops_and_waits():
    b = BeeClust(numpy.array([[5, 4, 0]]), p_wall=1, p_changedir=0)
    assert b.tick() == 0
    assert b.map[0, 0] == 5
    assert b.map[0, 2] == 0
    assert b.map[0, 1] <= -2  # waits at least min_wait
    assert b.map[0, 1] == -3  # actually waits based on temp


def test_hit_heater_stops_and_waits_longer():
    b = BeeClust(numpy.array([[6], [1], [0]]), p_wall=1, p_changedir=0)
    assert b.tick() == 0
    assert b.map[0, 0] == 6
    assert b.map[2, 0] == 0
    assert b.map[1, 0] <= -2
    assert b.map[1, 0] == -11  # warm, good, waits longer


def test_hit_strong_heater_stops_and_waits_very_long():
    b = BeeClust(numpy.array([[6], [1], [0]]),
                 p_wall=1, p_changedir=0, T_heater=100, T_ideal=90)
    assert b.tick() == 0
    assert b.map[0, 0] == 6
    assert b.map[2, 0] == 0
    assert b.map[1, 0] <= -2
    assert b.map[1, 0] == -15  # hot, very good, waits very long


def test_hit_cooler_stops_and_waits_shorter():
    b = BeeClust(numpy.array([[0], [3], [7]]), p_wall=1, p_changedir=0)
    assert b.tick() == 0
    assert b.map[0, 0] == 0
    assert b.map[2, 0] == 7
    assert b.map[1, 0] <= -2
    assert b.map[1, 0] == -2  # cold, bad, waits shorter


def test_hit_cooler_stops_and_waits_at_least_min():
    b = BeeClust(numpy.array([[0], [3], [7]]),
                 p_wall=1, p_changedir=0, min_wait=20)
    assert b.tick() == 0
    assert b.map[1, 0] == -20


def test_hit_cooler_stops_and_waits_longer():
    b = BeeClust(numpy.array([[0], [3], [7]]),
                 p_wall=1, p_changedir=0, T_cooler=-100, T_ideal=-90)
    assert b.tick() == 0
    assert b.map[0, 0] == 0
    assert b.map[2, 0] == 7
    assert b.map[1, 0] <= -2
    assert b.map[1, 0] == -15  # this kind of bee likes cold environment


def test_wall_may_stop_or_turn():
    # WARNING: This test is not deterministic, if it fails, run it again.
    # Let's not run another statistics and just check it works at least once.
    waited = False
    turned = False
    tries = 0
    while not turned and waited:
        tries += 1
        b = BeeClust(numpy.array([[0, 2, 5]]), p_wall=.5, p_changedir=0)
        assert b.tick() == 0
        if b.map[0, 1] == 4:
            turned = True
        else:
            assert b.map[0, 1] <= -2
            waited = True
        assert tries < 20


def test_two_bees_move_both():
    b = BeeClust(numpy.array([[0, 0, 0], [1, 0, 1]]), p_changedir=0)
    assert b.tick() == 2
    assert (b.map == [[1, 0, 1], [0, 0, 0]]).all()


def test_bee_hits_bee_keeps_trying():
    original = numpy.array([[0, 0, 2, 4, 0, 0]])
    b = BeeClust(original.copy(), p_changedir=0, p_meet=0)
    for _ in range(42):
        assert b.tick() == 0
        assert (b.map == original).all()


def test_bee_hits_bee_waits():
    b = BeeClust(numpy.array([[0, 0, 2, 4, 0, 0]]), p_changedir=0, p_meet=1)
    assert b.tick() == 0
    assert (b.map == [[0, 0, -3, -3, 0, 0]]).all()


def test_bee_hits_bee_waits_or_keeps_trying():
    b = BeeClust(numpy.array([[0, 0, 2, 4, 0, 0]]), p_changedir=0, p_meet=.5)
    assert b.tick() == 0
    assert b.map[0, 2] in (-3, 2)
    assert b.map[0, 3] in (-3, 4)
    # probability tests are left as an exercise for the reader


def test_forget_direction_waiting():
    for value in 1, 2, 3, 4, -2, -12, -3:
        b = BeeClust(loner(value), p_changedir=0)
        lo = loner(-1)
        b.map[0, 0] = lo[0, 0] = 5
        b.map[-1, -1] = lo[-1, -1] = 6
        b.map[0, -1] = lo[0, -1] = 7
        b.forget()
        assert (b.map == lo).all()
