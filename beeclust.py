import math
import numpy
import random
from collections import deque
from enum import Enum, IntEnum


class Map(IntEnum):
    """
    How are objects encoded in map.
    The IntEnum class is very useful here, we don't have to repeat .value
    when we access the underlying numbers.

    Negative numbers (not included here): A bee stopped for -N remaining ticks
    """
    EMPTY = 0
    BEE_NORTH = 1
    BEE_EAST = 2
    BEE_SOUTH = 3
    BEE_WEST = 4
    WALL = 5
    HEATER = 6
    COOLER = 7


class Movement(Enum):
    """
    Kinds of movement, only used by our algorithms, so no underlying numbers
    needed.
    """
    WALL_HIT = 0
    BEE_MEET = 1
    MOVE = 2
    WAIT = 3


DIR_OFFSETS_4 = {
    Map.BEE_NORTH: (-1, 0),
    Map.BEE_EAST: (0, 1),
    Map.BEE_SOUTH: (1, 0),
    Map.BEE_WEST: (0, -1)
}

DIR_OFFSETS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, 1), (-1, -1), (1, -1),
)


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

        self.map = map

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
        self.heatmap = numpy.empty(map.shape)
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

    @staticmethod
    def _is_bee(value):
        """
        Tells whether given value represents a bee (simple helper).

        value: Value to be bee-tested
        """
        return value < 0 or 1 <= value <= 4

    def tick(self):
        """
        Do single step of BeeClust algorithm.

        Returns number of moved bees.
        """
        done = numpy.full(self.map.shape, False, dtype='bool')
        moved = 0

        for r, c in numpy.ndindex(self.map.shape):
            if done[r, c]:
                continue
            if self.map[r, c] == -1:
                self.map[r, c] = random.randint(1, 4)
            elif 1 <= self.map[r, c] <= 4:
                if random.random() < self.p_changedir:
                    next_dir = random.randint(1, 3)
                    if next_dir == self.map[r, c]:
                        next_dir = 4
                    self.map[r, c] = next_dir

                offset_r, offset_c = DIR_OFFSETS_4[self.map[r, c]]
                nr = offset_r + r
                nc = offset_c + c

                movement = Movement.WALL_HIT
                if 0 <= nr < self.map.shape[0] and 0 <= nc < self.map.shape[1]:
                    if 1 <= self.map[nr, nc] <= 4 or self.map[nr, nc] < 0:
                        movement = Movement.BEE_MEET
                    elif self.map[nr, nc] == Map.EMPTY:
                        movement = Movement.MOVE

                if movement == Movement.WALL_HIT:
                    if random.random() < self.p_wall:
                        movement = Movement.WAIT
                    else:
                        self.map[r, c] = (self.map[r, c] + 1) % 4 + 1
                elif (movement == Movement.BEE_MEET
                      and random.random() < self.p_meet):
                    movement = Movement.WAIT

                if movement == Movement.WAIT:
                    delta = abs(self.heatmap[r, c] - self.T_ideal)
                    wait_time = int(self.k_stay / (1 + delta))
                    wait_time = max(self.min_wait, wait_time)
                    self.map[r, c] = -wait_time
                elif movement == Movement.MOVE:
                    moved += 1
                    self.map[nr, nc] = self.map[r, c]
                    self.map[r, c] = Map.EMPTY
                    done[nr, nc] = True
            elif self.map[r, c] < 0:
                self.map[r, c] += 1
            done[r, c] = True

        return moved

    def _inside(self, index):
        """
        Return whether the given 2D index is withing the boundaries
        of our map's shape.
        """
        shape = self.map.shape
        return 0 <= index[0] < shape[0] and 0 <= index[1] < shape[1]

    def _compute_distances(self, source):
        """
        Compute shortest distances to source in our map.

        The distance is computed omnidirectional (8 ways) using BFS.

        source: Source value (distance for such fields is 0)
        """
        indices = numpy.where(self.map == source)
        result = numpy.full(self.map.shape, -1, dtype='int64')
        sources = list(zip(indices[0], indices[1]))
        q = deque()
        for source_coords in sources:
            result[source_coords] = 0
            q.append((source_coords, 0))
        # BFS
        while q:
            (r, c), d = q.popleft()
            d += 1
            for offset_r, offset_c in DIR_OFFSETS_8:
                neighbor = r + offset_r, c + offset_c
                if self._inside(neighbor):
                    if self.map[neighbor] == Map.WALL:
                        continue
                    if result[neighbor] < 0 or result[neighbor] > d:
                        result[neighbor] = d
                        q.append((neighbor, d))
        return result

    def recalculate_heat(self):
        """
        Recalculate heat in BeeClust simulation

        This can be useful when you change the map. You are
        required to call this method on your own in the right
        time to ensure that simulation will be consistent.
        """
        heater_distances = self._compute_distances(Map.HEATER)
        cooler_distances = self._compute_distances(Map.COOLER)

        self.heatmap = numpy.full(self.map.shape, self.T_env, dtype='float')
        thk = math.fabs(self.T_heater - self.T_env)
        tck = math.fabs(self.T_cooler - self.T_env)
        for (r, c), value in numpy.ndenumerate(self.map):
            if value == Map.WALL:
                self.heatmap[r, c] = numpy.nan
            else:
                hd = heater_distances[r, c]
                cd = cooler_distances[r, c]
                if hd == 0:
                    self.heatmap[r, c] = self.T_heater
                elif cd == 0:
                    self.heatmap[r, c] = self.T_cooler
                else:
                    heating = (1 / hd) * thk
                    cooling = (1 / cd) * tck
                    delta = max(0, heating) - max(0, cooling)
                    self.heatmap[r, c] = self.T_env + self.k_temp * delta

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
        result = []
        shape = self.map.shape
        done = numpy.full(shape, False, dtype=bool)
        for pos, value in numpy.ndenumerate(self.map):
            if done[pos] or not self._is_bee(value):
                continue
            swarm = [pos]
            q = deque()
            q.append(pos)
            done[pos] = True
            # BFS
            while q:
                x, y = q.popleft()
                for offset_x, offset_y in DIR_OFFSETS_4.values():
                    neighbor = x + offset_x, y + offset_y
                    if self._inside(neighbor):
                        if self._is_bee(self.map[neighbor]):
                            if not done[neighbor]:
                                done[neighbor] = True
                                swarm.append(neighbor)
                                q.append(neighbor)
            result.append(swarm)
        return result

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
