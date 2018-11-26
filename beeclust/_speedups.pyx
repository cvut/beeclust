#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy
cimport numpy
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

# type shortcuts
from numpy cimport int64_t as int64
from numpy cimport float64_t as float64
from numpy cimport int8_t as int8
from numpy cimport uint8_t as uint8
from numpy cimport ndarray as nd


# Initialize random
srand(time(NULL))


cdef double rand_0_1():
    """
    Random number between 0 and 1
    """
    return <double>rand() / <double>RAND_MAX


cdef int randint(int limit):
    """
    Random positive integer up to {limit} (not included)
    """
    return rand() % limit


# A constant for further use by C code
cdef float NaN = float('nan')


# How are objects encoded in map.
# Negative numbers (not included here): A bee stopped for -N remaining ticks
cdef enum Map:
    EMPTY = 0
    BEE_NORTH = 1
    BEE_EAST = 2
    BEE_SOUTH = 3
    BEE_WEST = 4
    WALL = 5
    HEATER = 6
    COOLER = 7


# Kinds of movement.
cdef enum Movement:
    WALL_HIT = 0
    BEE_MEET = 1
    MOVE = 2
    WAIT = 3


# Coordinates / indices
cdef struct coords:
    int r
    int c


# A job to put into _JobQueue, has coordinates and distance
cdef struct job:
    coords loc
    int64 dist


cdef class _JobQueue:
    """
    Internal class with very basic (and quite stupid) queue.
    """
    cdef job * jobs
    cdef int top, bottom, size

    def __cinit__(self, int size):
        self.jobs = <job *>PyMem_Malloc(size * sizeof(job))
        if self.jobs == NULL:
            raise MemoryError()
        self.top = 0
        self.bottom = 0
        self.size = size

    def __dealloc__(self):
        if self.jobs != NULL:
            PyMem_Free(self.jobs)

    cdef void put(self, job ajob):
        self.jobs[self.top % self.size] = ajob
        self.top += 1

    cdef job get(self):
        self.bottom += 1
        return self.jobs[(self.bottom-1) % self.size]

    cdef bint empty(self):
        return self.bottom == self.top

    cdef void reset(self):
        self.top = 0
        self.bottom = 0


cdef nd[int64, ndim=2] compute_distances(nd[int8, ndim=2] m, int8 c):
    """
    Compute shortest distances to source in our map.
    The distance is computed omnidirectional (8 ways) using BFS.
    c: Source value (distance for such fields is 0)
    """
    cdef nd[int64, ndim=2] result
    cdef _JobQueue jobs = _JobQueue(m.size * 8)
    cdef int i
    cdef coords loc
    cdef job ajob
    cdef coords nloc
    cdef int r_off, c_off
    cdef int64 dist

    result = numpy.full((m.shape[0], m.shape[1]), -1, dtype='int64')
    indices = numpy.where(m == c)

    for i in range(len(indices[0])):
        loc = coords(indices[0][i], indices[1][i])
        result[loc.r, loc.c] = 0
        jobs.put(job(loc, 0))

    # BFS
    while not jobs.empty():
        ajob = jobs.get()
        loc = ajob.loc
        dist = ajob.dist + 1
        for r_off in range(-1, 2):
            for c_off in range(-1, 2):
                if (r_off == 0) and (c_off == 0):
                    continue
                nloc.r = loc.r + r_off
                nloc.c = loc.c + c_off
                if 0 <= nloc.r < m.shape[0] and 0 <= nloc.c < m.shape[1]:
                    if m[nloc.r, nloc.c] == WALL:
                        continue
                    if (result[nloc.r, nloc.c] < 0 or
                            result[nloc.r, nloc.c] > dist):
                        result[nloc.r, nloc.c] = dist
                        jobs.put(job(nloc, dist))
    return result


def recalculate_heat(nd[float64, ndim=2] heatmap, nd[int8, ndim=2] m,
                     float64 T_heater, float64 T_cooler,
                     float64 T_env, float64 k_temp):
    """
    Fast implementation for BeeClust.recalculate_heat().
    """
    cdef float64 heating, cooling
    cdef int r, c
    cdef int64 hd, cd
    cdef float64 thk = T_heater - T_env
    cdef float64 tck = T_env - T_cooler

    cdef nd[int64, ndim=2] heater_distances = compute_distances(m, HEATER)
    cdef nd[int64, ndim=2] cooler_distances = compute_distances(m, COOLER)

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if m[r, c] == WALL:
                heatmap[r, c] = NaN
            else:
                hd = heater_distances[r, c]
                cd = cooler_distances[r, c]
                if hd == 0:
                    heatmap[r, c] = T_heater
                elif cd == 0:
                    heatmap[r, c] = T_cooler
                else:
                    heating = (1.0 / hd) * thk
                    cooling = (1.0 / cd) * tck
                    heatmap[r, c] = (T_env + k_temp *
                                     (max(0, heating) - max(0, cooling)))
    return heatmap


cdef bint _is_bee(int8 value):
    """
    Tells whether given value represents a bee (simple helper).
    value: Value to be bee-tested
    """
    return value < 0 or 1 <= value <= 4


def swarms(nd[int8, ndim=2] m):
    """
    Fast implementation for BeeClust.swarms.
    """
    cdef nd[uint8, ndim=2] done
    cdef coords loc, nloc
    cdef job ajob
    cdef int direction
    # we don't use the .dist attr of the job, but meh, let's reuse the queue
    cdef _JobQueue q = _JobQueue(m.size)

    done = numpy.zeros((m.shape[0], m.shape[1]), dtype='uint8')
    result = []

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if done[r, c] or not _is_bee(m[r, c]):
                continue
            swarm = [(r, c)]
            q.reset()
            q.put(job(coords(r, c), 0))
            done[r, c] = True
            # BFS
            while not q.empty():
                ajob = q.get()
                loc = ajob.loc
                for direction in range(4):
                    nloc = loc
                    if direction == 0:
                        nloc.r -= 1
                    elif direction == 1:
                        nloc.c += 1
                    elif direction == 2:
                        nloc.r += 1
                    else:
                        nloc.c -= 1
                    if 0 <= nloc.r < m.shape[0] and 0 <= nloc.c < m.shape[1]:
                        if (_is_bee(m[nloc.r, nloc.c])
                                and not done[nloc.r, nloc.c]):
                            done[nloc.r, nloc.c] = True
                            swarm.append((nloc.r, nloc.c))
                            q.put(job(nloc, 0))
            result.append(swarm)
    return result


def tick(nd[int8, ndim=2] m,
         nd[float64, ndim=2] heatmap,
         double p_changedir, double p_wall, double p_meet,
         int8 min_wait, double k_stay, double T_ideal):
    """
    Fast implementation for BeeClust.tick().
    """
    cdef nd[uint8, ndim=2] done
    cdef int r, c
    cdef int8 next_dir, wait
    cdef int moved = 0

    done = numpy.zeros((m.shape[0], m.shape[1]), dtype='uint8')

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            if done[r, c]:
                continue
            if m[r, c] == -1:
                m[r, c] = randint(4) + 1
            elif 1 <= m[r, c] <= 4:
                if rand_0_1() < p_changedir:
                    next_dir = randint(3) + 1
                    if next_dir == m[r, c]:
                        next_dir = 4
                    m[r, c] = next_dir

                if m[r, c] == BEE_NORTH:
                    nr, nc = r - 1, c
                elif m[r, c] == BEE_EAST:
                    nr, nc = r, c + 1
                elif m[r, c] == BEE_SOUTH:
                    nr, nc = r + 1, c
                else:  # BEE_WEST
                    nr, nc = r, c - 1

                movement = WALL_HIT
                if 0 <= nr < m.shape[0] and 0 <= nc < m.shape[1]:
                    if 1 <= m[nr, nc] <= 4 or m[nr, nc] < 0:
                        movement = BEE_MEET
                    elif m[nr, nc] == EMPTY:
                        movement = MOVE

                if movement == WALL_HIT:
                    if rand_0_1() < p_wall:
                        movement = WAIT
                    else:
                        m[r, c] = (m[r, c] + 1) % 4 + 1
                elif movement == BEE_MEET and rand_0_1() < p_meet:
                    movement = WAIT

                if movement == WAIT:
                    wait = <int8>(k_stay /(1 + abs(heatmap[r, c] - T_ideal)))
                    wait = max(min_wait, wait)
                    m[r, c] = -1 * wait
                elif movement == MOVE:
                    moved += 1
                    m[nr, nc] = m[r, c]
                    m[r, c] = EMPTY
                    done[nr, nc] = True
            elif m[r, c] < 0:
                m[r, c] += 1
            done[r, c] = True

    return moved
