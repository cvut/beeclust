import numpy

from . import _speedups


class BeeClust:
    """
    BeeClust swarming algorithm simulation.

    Arguments:
     map (required): a 2D numpy-like array of numbers of objects (see Map)
     p_changedir: probability of a bee changing it's direction
     p_wall: probability of a bee stopping when it hits a wall
     p_meet: probability of a bee stopping when it hits another bee
     k_temp: coefficient for thermal conductivity
     k_stay: coefficient of staying (larger -> bees remain stopped longer)
     T_ideal, T_heater, T_cooler, T_env:
       ideal temperature for bees, temperature of heaters, coolers, environment
     min_wait: minimal time a bee remains stopped

    Attributes:
     map: the actual map as described above
     heatmap: information about temperature of every field of the map
     bees: list of tuples (indices) of bees locations
     swarm: list of lists of tuples (indices) with connecting bees
     score: average temperature of fields with bees
    """
    def __init__(self, map,
                 p_changedir=0.2, p_wall=0.8, p_meet=0.8,
                 k_temp=0.9, k_stay=50,
                 T_ideal=35, T_heater=40, T_cooler=5, T_env=22,
                 min_wait=2):
        try:
            if map.ndim != 2:
                raise ValueError(
                    f'Wrong number of dimensions ({map.ndim}, expected 2)')
        except AttributeError:
            raise TypeError('Wrong type of map, it has no .ndim.')

        self.map = map.astype(numpy.int8)

        self._set_numeric('p_changedir', p_changedir)
        self._set_numeric('p_wall', p_wall)
        self._set_numeric('p_meet', p_meet)

        self._set_numeric('k_temp', k_temp)
        self._set_numeric('k_stay', k_stay)

        self._set_numeric('T_ideal', T_ideal, neg=True)
        self._set_numeric('T_heater', T_heater, neg=True)
        self._set_numeric('T_cooler', T_cooler, neg=True)
        self._set_numeric('T_env', T_env, neg=True)

        self._set_numeric('min_wait', min_wait)

        if not (T_cooler <= T_env <= T_heater):
            raise ValueError('Make sure that T_cooler <= T_env <= T_heater')

        self._heater_distances = numpy.empty(map.shape)
        self._cooler_distances = numpy.empty(map.shape)
        self.recalculate_heat()

    def _set_numeric(self, name, value, *, neg=False):
        """
        Set numeric attribute of self with constraints.

        It checks whether the value is int or float and optionally
        if is only positive. By name it checks also that probabilities
        are not larger than 1.

        Note that we don't check for any numerical type, because e.g. complex
        numbers won't work.

        name: name of attribute to be set
        value: (numerical) value
        neg: if the value can be negative
        """
        if isinstance(value, (int, float)):
            if value < 0 and not neg:
                raise ValueError(f'{name} cannot be negative')
            if name.startswith('p_') and value > 1:
                raise ValueError(
                    f'{name} is a probability, it cannot be larger than 1')
            setattr(self, name, value)
        else:
            raise TypeError(f'Wrong type of {name}: {type(value).__name__}')

    def tick(self):
        """
        Do single step of BeeClust algorithm.

        Returns number of moved bees.
        """
        return _speedups.tick(self.map, self.heatmap, self.p_changedir,
                              self.p_wall, self.p_meet, self.min_wait,
                              self.k_stay, self.T_ideal)

    def recalculate_heat(self):
        """
        Recalculate heat in BeeClust simulation

        This can be useful when you change the map. You are
        required to call this method on your own in the right
        time to ensure that simulation will be consistent.
        """
        heatmap = numpy.full(self.map.shape, self.T_env, dtype='float64')
        self.heatmap = _speedups.recalculate_heat(heatmap, self.map,
                                                  self.T_heater, self.T_cooler,
                                                  self.T_env, self.k_temp)

    @property
    def bees(self):
        """
        Enlist coordinates where bees are located
        """
        indices = numpy.where((self.map < 0)
                              | ((1 <= self.map) & (self.map <= 4)))
        return list(zip(indices[0], indices[1]))

    @property
    def swarms(self):
        """
        Enlist swarms as lists of coords of bees
        """
        return _speedups.swarms(self.map)

    @property
    def score(self):
        """
        Compute score as average bee's temperature
        """
        temps = [self.heatmap[pos] for pos in self.bees]
        if len(temps) == 0:
            raise ValueError('No bees in beeclust')
        return sum(temps) / len(temps)

    def forget(self):
        """
        Make all bees to forget their movement direction
        """
        self.map[numpy.where((self.map < 0)
                             | ((1 <= self.map) & (self.map <= 4)))] = -1
