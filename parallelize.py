#!/usr/bin/env python

import argparse
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
import dask
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from photutils import datasets
from photutils import DAOStarFinder
from photutils import CircularAperture
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dask',
                    help='Use dask instead of multiprocessing (True)',
                    action='store_true',
                    default=False)

parser.add_argument('-n',
                    help='Number of datasets to process (20)',
                    default=20,
                    type=int)

parser.add_argument('-threads',
                    help='Use threads instead of processes with dask (False)',
                    default=False,
                    action='store_true')

parser.add_argument('-nworkers',
                    help='set the number of workers (os.cpu_count())',
                    default=os.cpu_count(),
                    type=int)

parser.add_argument('-s', help='Run the analysis serially (False)',
                    default=False, action='store_true')

class FindStars(object):
    """docstring for FindStars"""
    def __init__(self, data):

        self.data = data
        self.sources = None

    def find(self):
        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0, iters=5)
        daofind = DAOStarFinder(fwhm=5.5, threshold=5*std)
        self.sources = daofind(self.data - median)

    def show_stars(self, data, sources):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        norm = ImageNormalize(data,
                              stretch=LinearStretch(),
                              interval=ZScaleInterval())
        im = ax.imshow(data, norm=norm, cmap='gray')
        # Make apertures
        apertures = CircularAperture((sources['xcentroid'],
                                      sources['ycentroid']),r=3)
        apertures.plot(color='red', lw=1.5, alpha=0.75)
        plt.show()

class Parallelize(FindStars):
    """
    Small class for holding data and running the analysis
    """
    def __init__(self, dataset):
        super(FindStars, self).__init__()
        self.dataset = dataset
        self.N = len(dataset)
        self.sources = []
        self.num_sources = None

    def analyze(self, data):
        obj = FindStars(data)
        obj.find()

        return obj.sources

    def run_serial(self):
        """ Find sources one image at a time

        Returns
        -------
        total processing time

        """
        start_time = time.time()
        for dset in self.dataset:
            self.sources.append(self.analyze(dset))
        end_time = time.time()
        runtime = (end_time - start_time) / 60
        self.num_sources = sum([len(val) for val in self.sources])
        print('Total number of sources {}'.format(self.num_sources))
        print('Processing Time: {:.4f} minutes'.format(runtime))

        return runtime

    def run_parallel(self, threads=False, use_dask=True, nworkers=None):
        """ Find sources in all images at once

        This will spawn a total of nworkers processes to use in source
        identification across all images at once.

        Parameters
        ----------
        threads : Boolean flag for using threads instead of processes
        use_dask : Boolean flag for using dask instead of multiprocessing module
        nworkers : number of processes to spawn

        Returns
        -------
        total processing time
        """

        if use_dask:
            print('Analyzing {} datasets with dask'.format(self.N))
            start_time = time.time()
            delayed_objects = [dask.delayed(self.analyze)(dset)
                               for dset in self.dataset]
            if threads:
                print('Multithreading')
                self.sources = dask.compute(*delayed_objects,
                                       scheduler='threads',
                                       num_workers=nworkers)
            else:
                print('Multiprocessing')
                self.sources = dask.compute(*delayed_objects,
                                       scheduler='processes',
                                       num_workers=nworkers)
            end_time = time.time()
        else:
            print('Analyzing {} datasets with multiprocessing'.format(self.N))
            start_time = time.time()
            with mp.Pool(nworkers) as pool:
                self.sources = pool.map(self.analyze,self.dataset)
            end_time = time.time()

        runtime = (end_time - start_time)/60
        self.num_sources = sum([len(val) for val in self.sources])
        print('Total number of sources {}'.format(self.num_sources))
        print('Processing Time: {:.4f} minutes'.format(runtime))
        return runtime

    def plot(self):
        self.show_stars(self.dataset[0], self.sources[0])



def main(N=20, serial=False, use_dask=False,
         dask_threads=False, nworkers=os.cpu_count()):

    hdu = datasets.load_star_image()
    dataset = [hdu.data]*N
    obj = Parallelize(dataset)
    if not serial:
        obj.run_parallel(use_dask=use_dask,
                         threads=dask_threads,
                         nworkers=nworkers)
    else:
        obj.run_serial()

    return obj

if __name__ == '__main__':
    args = parser.parse_args()
    obj = main(args.N, args.s, args.dask, args.threads, args.nworkers)