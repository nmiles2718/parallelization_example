#!/usr/bin/env python
import dask
from multiprocessing import Pool
import time


def combine(a=0, b=0, c=0):
    """ Simple function to compute the sum of three numbers

    Emulate a more complicated task by telling the computer to wait for 0.5
    seconds before computing and returning the sum of a, b, and c.

    Parameters
    ----------
    a
    b
    c

    Returns
    -------
    a + b + c
    """
    time.sleep(0.5)
    return a + b + c


def run_serial():
    """ Run the computation serially to establish a baseline

    Returns
    -------
    results : list of results containing the sums of the 3 input numbers
    runtime : the time is took to complete the task
    """
    # Get the start time
    st = time.time()

    # Convenience function for generating a list of monotonically increasing
    # numbers starting from start and ending at end.
    f = lambda start, end: [i for i in range(start, end, 1)]

    # Generate three separate lists of 100 numbers
    a = f(1, 10)
    b = f(10, 21)
    c = f(20, 31)

    # Combine the three lists to create a list of 100 tuples of three numbers
    # i.e. [(a1, b1, c1), (a2, b2, c2),..., (a100, b100, c100)]
    inputs = list(zip(a, b, c))
    results= []
    for arg in inputs:
        results.append(combine(*arg))

    # Get the end time and compute the total runtime
    et = time.time()
    runtime = et - st
    return results, runtime


def use_multiprocessing():
    """Parallelize the summing 100 pairs of 3 numbers with multiprocessing

    Returns
    -------
    results : list of results containing the sums of the 3 input numbers
    runtime : the time is took to complete the task
    """
    # Get the start time
    st = time.time()

    # Convenience function for generating a list of monotonically increasing
    # numbers starting from start and ending at end.
    f = lambda start, end: [i for i in range(start, end, 1)]

    # Generate three separate lists of 100 numbers
    a = f(1, 10)
    b = f(10, 21)
    c = f(20, 31)

    # Combine the three lists to create a list of 100 tuples of three numbers
    # i.e. [(a1, b1, c1), (a2, b2, c2),..., (a100, b100, c100)]
    inputs = list(zip(a, b, c))

    # Create a pool of 8 processes and map the combine function across all inputs
    with Pool(8) as p:
        results = p.starmap(combine, inputs)

    # Get the end time and compute the total runtime
    et = time.time()
    runtime = et - st
    return results, runtime


def use_dask(scheduler='processes'):
    """Parallelize the summing of 100 pairs of 3 numbers with dask

    Returns
    -------
    results : list of results containing the sums of the 3 input numbers
    runtime : the time is took to complete the task
    """
    # Get the start time
    st = time.time()

    # Convenience function for generating a list of monotonically increasing
    # numbers starting from start and ending at end.
    f = lambda start, end: [i for i in range(start, end, 1)]

    # Generate three separate lists of 100 numbers
    a = f(1, 10)
    b = f(10, 21)
    c = f(20, 31)

    # Combine the three lists to create a list of 100 tuples of three numbers
    # i.e. [(a1, b1, c1), (a2, b2, c2),..., (a100, b100, c100)]
    inputs = list(zip(a, b, c))

    # Generate a list of delayed objects
    results = [dask.delayed(combine)(a=arg[0], b=arg[1], c=arg[2]) for arg in
               inputs]

    # Tell dask to execute the delayed objects using 8 processes
    results = list(dask.compute(*results,
                                num_workers=8,
                                scheduler=scheduler))

    # Get the time at the end and compute the total runtime
    et = time.time()
    runtime = et - st
    return results, runtime


def compare():
    # Run trivial example using multiprocessing
    mp_results, mp_time = use_multiprocessing()

    # Run trivial example using dask
    dask_results, dask_time = use_dask()

    # Run trivial example serially
    serial_results, serial_time = run_serial()

    # Check to see that each package returns the same results
    # If they do, compare the times
    try:
        assert dask_results == mp_results
        assert serial_results == dask_results
    except AssertionError:
        print('Results are different!')
        print('This should never happen!')
    else:
        print('Time to compute:')
        print('serially: {:.4f}'.format(serial_time))
        print('dask: {:.4f}s'.format(dask_time))
        print('multiprocessing: {:.4f}s'.format(mp_time))


if __name__ == '__main__':
    compare()
