import numpy


def zeros8(*args, **kwargs):
    kwargs.setdefault('dtype', numpy.int8)
    return numpy.zeros(*args, **kwargs)


def full8(*args, **kwargs):
    kwargs.setdefault('dtype', numpy.int8)
    return numpy.full(*args, **kwargs)
