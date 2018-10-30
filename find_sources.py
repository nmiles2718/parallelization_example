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

parser.add_argument('-plot', help='Show an example of the sources found',
                    action='store_true', default=False)


def find_sources(data, sigma_clip_thresh=3, fwhm=5.5, threshold=5):
    """ Method for finding sources in the input data

    Parameters
    ----------
    sigma_clip_thresh : threshold to use in sigma clipping
    fwhm : FWHM of gaussian kernel to use in photutils.DAOStarFinder()
    threshold : threshold above background to use in source detection

    Returns
    -------

    """
    print(fwhm)
    mean, median, std = sigma_clipped_stats(data,
                                            sigma=sigma_clip_thresh,
                                            iters=5)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median)
    return sources


def show_sources(data, sources):
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
    im = ax.imshow(data, norm=norm, cmap='gray', origin='lower')
    # Make apertures
    apertures = CircularAperture((sources['xcentroid'],
                                  sources['ycentroid']),r=3)
    apertures.plot(color='red', lw=1.5, alpha=0.75)
    ax.grid('off')
    fig.savefig('example_image.png', format='png', dpi=275)
    plt.show()


def run_serial(dataset):
    """ Find sources one image at a time

    Parameters
    ----------
    dataset : list of FITs data to analyze

    Returns
    -------
    total processing time

    """
    sources = []
    start_time = time.time()
    for dset in dataset:
        sources.append(find_sources(dset,
                                sigma_clip_thresh=3,
                                fwhm=5.5,
                                threshold=5))
    end_time = time.time()
    runtime = (end_time - start_time) / 60
    num_sources = sum([len(val) for val in sources])
    print('Total number of sources {}'.format(num_sources))
    print('Processing Time: {:.4f} minutes'.format(runtime))
    return runtime, sources


def run_parallel(dataset, use_dask=True, nworkers=None):
    """ Find sources in all images at once

    This will spawn a total of nworkers processes to use in source
    identification across all images in a parallelized manner.

    Parameters
    ----------
    dataset : List of FITS image to parallelize over
    use_dask : Boolean flag for using dask instead of multiprocessing module
    nworkers : number of processes to spawn

    Returns
    -------
    total processing time
    """

    N = len(dataset) # number of images

    if use_dask:
        print('Analyzing {} datasets with dask'.format(N))
        start_time = time.time()
        delayed_objects = [dask.delayed(find_sources)(dset,
                                                      sigma_clip_thresh=3,
                                                      fwhm=5.5,
                                                      threshold=5)
                           for dset in dataset]

        sources = dask.compute(*delayed_objects,
                                    scheduler='processes',
                                    num_workers=nworkers)
        end_time = time.time()
    else:
        print('Analyzing {} datasets with multiprocessing'.format(N))
        start_time = time.time()
        # Because of the limitations with multiprocessing, we have to
        # generate lists of all the constant values that are the same
        # length as our list of the datasets
        fwhm = [3]*N
        sigma_clipping_thresh = [3]*N
        source_finding_thresh = [5]*N
        inputs = list(zip(dataset,
                          sigma_clipping_thresh,
                          fwhm,
                          source_finding_thresh))
        with mp.Pool(nworkers) as pool:
            sources = pool.starmap(find_sources, inputs)
        end_time = time.time()

    runtime = (end_time - start_time) / 60
    num_sources = sum([len(val) for val in sources])
    print('Total number of sources {}'.format(num_sources))
    print('Processing Time: {:.4f} minutes'.format(runtime))
    return runtime, sources


def generate_dataset(N):
    """ Use the photutils.dataset module to load in some example data.

    Parameters
    ----------
    N : Number of images to generate for this dataset

    Returns
    -------
    A list of FITS images
    """
    hdu = datasets.load_star_image()
    dataset = [hdu.data] * N
    return dataset


def analyze(N=20, serial=False, use_dask=False, nworkers=8, plot=True):
    """

    Parameters
    ----------
    N : Number of fake datasets to generate
    serial : Boolean flag for running serially or in parallel
    use_dask : Boolean flag for using dask instead of multiprocessing
    nworkers : Number of work processes to spawn (i.e. cores to use)

    Returns
    -------

    """
    # Generate a sample dataset using the photutils datasets
    dataset = generate_dataset(N)

    if not serial:
       runtime, sources = run_parallel(dataset, use_dask=use_dask, nworkers=nworkers)
    else:
        runtime, sources = run_serial(dataset)

    if plot:
        show_sources(dataset[0], sources[0])
    num_sources = sum([len(val) for val in sources])

    return runtime, num_sources

if __name__ == '__main__':
    args = parser.parse_args()
    runtime, num_sources = analyze(args.n, args.s, args.dask,
                                   args.nworkers, args.plot)