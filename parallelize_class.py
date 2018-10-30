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

parser.add_argument('-nworkers',
                    help='set the number of workers (os.cpu_count())',
                    default=os.cpu_count(),
                    type=int)

parser.add_argument('-s', help='Run the analysis serially (False)',
                    default=False, action='store_true')

class FindStars(object):
    """
    A simple class for finding stars. It has two methods; find and show_stars

    The find()
    """
    def __init__(self, data):

        self.data = data
        self.sources = None

    def find(self, sigma_clip_thresh=3, fwhm=5.5, threshold=5):
        """ Method for finding sources in the input data

        Parameters
        ----------
        sigma_clip_thresh : threshold to use in sigma clipping
        fwhm : FWHM of gaussian kernel to use in photutils.DAOStarFinder()
        threshold : threshold above background to use in source detection

        Returns
        -------

        """
        mean, median, std = sigma_clipped_stats(self.data,
                                                sigma=sigma_clip_thresh,
                                                iters=5)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        self.sources = daofind(self.data - median)

    def show_stars(self, data, sources):
        """ Convenience method for plotting the identified sources

        Parameters
        ----------
        data : FITS data to plot
        sources : Sources found in the FITS data

        Returns
        -------

        """

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

    def analyze(self, data, sigma_clip_thresh=3, fwhm=5.5, threshold=5):
        """ Find sources in the data

        Parameters
        ----------
        data : FITS data
        sigma_clip_thresh : threshold to use in sigma clipping
        fwhm : FWHM of gaussian kernel to use in photutils.DAOStarFinder()
        threshold : threshold above background to use in source detection

        Returns
        -------
        All sources identified in input image
        """
        # Instantiate a FindStars object
        obj = FindStars(data)

        # Use the find method on the FindStars object to identify sources
        obj.find(sigma_clip_thresh, fwhm, threshold)
        return obj.sources

    def run_serial(self):
        """ Find sources one image at a time

        Returns
        -------
        total processing time

        """
        tmp = []
        start_time = time.time()
        for dset in self.dataset:
            tmp.append(self.analyze(dset,
                                    sigma_clip_thresh=3,
                                    fwhm=5.5,
                                    threshold=5))
        self.sources = tmp
        end_time = time.time()
        runtime = (end_time - start_time) / 60
        self.num_sources = sum([len(val) for val in self.sources])
        print('Total number of sources {}'.format(self.num_sources))
        print('Processing Time: {:.4f} minutes'.format(runtime))

        return runtime

    def run_parallel(self, use_dask=True, nworkers=None):
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
            # Isn't dask nice?
            delayed_objects = [dask.delayed(self.analyze)(dset,
                                                          sigma_clipping=3,
                                                          fwhm=5.5,
                                                          threshold=5)
                               for dset in self.dataset]

            self.sources = dask.compute(*delayed_objects,
                                        scheduler='processes',
                                        num_workers=nworkers)
            end_time = time.time()
        else:
            print('Analyzing {} datasets with multiprocessing'.format(self.N))
            start_time = time.time()
            # Because of the limitations with multiprocessing, we have to
            # generate lists of all the constant values that are the same
            # length as our list of the datasets
            fwhm = [5.5]*self.N
            sigma_clipping_thresh = [3]*self.N
            source_finding_thresh = [5]*self.N
            inputs = list(zip(self.dataset,
                              sigma_clipping_thresh,
                              fwhm,
                              source_finding_thresh))
            with mp.Pool(nworkers) as pool:
                self.sources = pool.starmap(self.analyze, inputs)
            end_time = time.time()

        runtime = (end_time - start_time)/60
        self.num_sources = sum([len(val) for val in self.sources])
        print('Total number of sources {}'.format(self.num_sources))
        print('Processing Time: {:.4f} minutes'.format(runtime))
        return runtime

    def plot(self):
        self.show_stars(self.dataset[0], self.sources[0])



def main(N=20, serial=False, use_dask=False, nworkers=os.cpu_count()):

    hdu = datasets.load_star_image()
    dataset = [hdu.data]*N
    obj = Parallelize(dataset)
    if not serial:
        obj.run_parallel(use_dask=use_dask,
                         nworkers=nworkers)
    else:
        obj.run_serial()

    return obj

if __name__ == '__main__':
    args = parser.parse_args()
    obj = main(args.n, args.s, args.dask, args.nworkers)