import pytest
from collections import OrderedDict
from helpers import zeros8
from beeclust import BeeClust


MAP = zeros8((3, 2))
KWARGS = OrderedDict(
    map=MAP,
    p_changedir=0.2,
    p_wall=0.8,
    p_meet=0.8,
    k_temp=0.9,
    k_stay=50,
    T_ideal=35,
    T_heater=40,
    T_cooler=5,
    T_env=22,
    min_wait=2,
)


def test_init_all():
    BeeClust(**KWARGS)


def test_init_all_postional():
    BeeClust(*KWARGS.values())


def test_optional_kwargs():
    for key in KWARGS.keys():
        if key == 'map':
            continue
        kwargs = OrderedDict(**KWARGS)
        del kwargs[key]
        BeeClust(**kwargs)


def test_individual_kwargs():
    for key in KWARGS.keys():
        if key == 'map':
            continue
        kwargs = {key: KWARGS[key]}
        BeeClust(MAP, **kwargs)


def test_str_kwargs_raise_TypeError():
    for key in KWARGS.keys():
        kwargs = OrderedDict(**KWARGS)
        kwargs[key] = 'impossibru'
        with pytest.raises(TypeError) as excinfo:
            BeeClust(**kwargs)
        assert key in str(excinfo.value)


def test_weird_map_shape_raises_ValueError():
    for shape in (8,), (2, 2, 2):
        kwargs = OrderedDict(**KWARGS)
        kwargs['map'] = zeros8(shape)
        with pytest.raises(ValueError) as excinfo:
            BeeClust(**kwargs)
        assert ('shape' in str(excinfo.value) or 'dim' in str(excinfo.value))


def test_negative_values_raise_ValueError():
    for key in KWARGS.keys():
        if key == 'map' or key.startswith('T_'):
            continue
        kwargs = OrderedDict(**KWARGS)
        kwargs[key] *= -1
        with pytest.raises(ValueError) as excinfo:
            BeeClust(**kwargs)
        assert ('negative' in str(excinfo.value) or
                'positive' in str(excinfo.value))


def test_invalid_probability_values_raise_ValueError():
    for key in 'p_changedir', 'p_wall', 'p_meet':
        kwargs = OrderedDict(**KWARGS)
        kwargs[key] = 2
        with pytest.raises(ValueError) as excinfo:
            BeeClust(**kwargs)
        assert ('probability' in str(excinfo.value) and
                '1' in str(excinfo.value))


def test_cold_heater():
    with pytest.raises(ValueError) as excinfo:
        BeeClust(
            MAP,
            T_cooler=2,
            T_env=10,
            T_heater=8,
        )
    assert ('T_env' in str(excinfo.value) or 'T_heater' in str(excinfo.value))


def test_hot_cooler():
    with pytest.raises(ValueError) as excinfo:
        BeeClust(
            MAP,
            T_cooler=12,
            T_env=10,
            T_heater=20,
        )
    assert ('T_env' in str(excinfo.value) or 'T_cooler' in str(excinfo.value))


def test_inert_temps():
    # shouldn't rise
    BeeClust(
        MAP,
        T_cooler=10,
        T_env=10,
        T_heater=10,
    )


def test_negative_temps():
    BeeClust(
        MAP,
        T_cooler=-10,
        T_env=1,
        T_heater=5,
    )
