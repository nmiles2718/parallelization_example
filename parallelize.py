#!/usr/bin/env python

import argparse
import dask
from findstars import FindStars
from photutils import datasets
import multiprocessing as mp
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('-dask',
                    help='Use dask instead of multiprocessing (default=True)',
                    action='store_true',
                    default=False)

parser.add_argument('-N', help='Number of datasets to process (default=20)',
                    default=20, type=int)

parser.add_argument('-threads', help='If using dask, this will parallelize'
                                     ' with threads instead of processes',
                    default=False, action='store_true')

class Parallel(object):
    """
    Small class for hold dataing and running the analysis
    """
    def __init__(self, dataset):
        self.dataset = dataset


    def analyze(self, data):
        obj = FindStars(data)
        obj.run_analysis()
        return obj.sources

    def run_serial(self):
        sources = []
        total = 0
        start_time = time.time()
        for dset in self.dataset:
            sources.append(self.analyze(dset))
        end_time = time.time()
        total = (end_time - start_time) / 60
        print('Total number of sources {}'.format(sum([len(val)
                                                       for val in sources])))
        print('Processing Time: {:.4f} minutes'.format(total))
        return total, sources

    def run_parallel(self, threads=False, use_dask=True, n_cores=None):
        if use_dask:
            start_time = time.time()
            delayed_objects = [dask.delayed(self.analyze)(dset) for dset in self.dataset]
            if threads:
                print('Multithreading')
                sources = dask.compute(*delayed_objects,
                                       scheduler='threads',
                                       num_workers=n_cores)
            else:
                print('Multiprocessing')
                sources = dask.compute(*delayed_objects,
                                       scheduler='processes',
                                       num_workers=n_cores)
            end_time = time.time()
        else:
            start_time = time.time()
            with mp.Pool(n_cores) as pool:
                sources = pool.map(self.analyze,self.dataset)
            end_time = time.time()
        total = (end_time - start_time)/60
        print('Total number of sources {}'.format(sum([len(val)
                                                       for val in sources])))
        print('Processing Time: {:.4f} minutes'.format(total))
        return total, sources



def main(N=20, use_dask=False, dask_threads=False, n_cores = os.cpu_count()):
    if use_dask:
        print('Analyzing {} datasets with dask'.format(N))
    else:
        print('Analyzing {} datasets with multiprocessing'.format(N))
    hdu = datasets.load_star_image()
    dataset = [hdu.data]*N
    obj = Parallel(dataset)
    obj.run_parallel(use_dask=use_dask, threads=dask_threads, n_cores=n_cores)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.N, args.dask, args.threads)