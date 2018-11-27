import numpy
import pytest

from beeclust import BeeClust

# if you PC is not strong enough, you might consider lowering this temporarily
SIZE = 1024


def random_map():
    p = [.35, .05, .05, .05, .05, .05, .2, .2]
    return numpy.random.choice(len(p), SIZE ** 2, p=p).reshape((SIZE, SIZE))


# we can make 20 runs in 5 s, so we give you 10 s
@pytest.mark.timeout(10)
def test_heatmap_is_fast():
    for i in range(20):
        print(i)
        b = BeeClust(random_map())
        assert b.heatmap.shape


# prepare a BeeClust outside of a test
a_beeclust = BeeClust(random_map())
# touch the heatmap just for sure
a_beeclust.heatmap.shape


@pytest.mark.timeout(10)
def test_swarms_is_fast():
    swarms = None
    for i in range(20):
        print(i)
        numpy.random.shuffle(a_beeclust.map)
        assert a_beeclust.swarms != swarms
        swarms = a_beeclust.swarms


# prepare a BeeClust outside of a test
b_beeclust = BeeClust(random_map())
# touch the heatmap just for sure
b_beeclust.heatmap.shape


@pytest.mark.timeout(10)
def test_tick_is_fast():
    moved = 0
    for i in range(20):
        print(i)
        nmoved = b_beeclust.tick()
        assert nmoved != moved  # well, it could, but low probability
        moved = nmoved
