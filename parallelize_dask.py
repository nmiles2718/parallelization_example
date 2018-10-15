
import argparse

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
import dask
import matplotlib.pyplot as plt
import numpy as np
from photutils import DAOStarFinder
from photutils import datasets
from photutils import CircularAperture
import time


parser = argparse.ArgumentParser()
parser.add_argument('-p', help='parallelize or not', 
                    action='store_true', default=False)

parser.add_argument('-n', help='number of datasets',
                     type=int, default=10)

parser.add_argument('-t', help='use threads',
                    action='store_true', default=False)

parser.add_argument('-f', help='use file', type=str, default=None)

class FindStars(object):
    """docstring for FindStars"""
    def __init__(self, data, f):
        super(FindStars, self).__init__()
        self.data = data
        self.sources = None
        self.fname = f

    def grab_data(self):
        with fits.open(self.fname) as hdu:
            self.data = hdu[1].data

    def find(self):
        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0, iters=5)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std) 
        self.sources = daofind(self.data - median)

    def show_stars(self):
        norm = ImageNormalize(self.data, stretch=SqrtStretch(), interval=ZScaleInterval())
        im = plt.imshow(self.data, norm=norm, cmap='gray')

        # Make apertures
        apertures = CircularAperture((self.sources['xcentroid'], self.sources['ycentroid']),r=5)
        apertures.plot(color='red', lw=1.5, alpha=0.75)
        plt.show()

    def run_analysis(self):
        if not self.fname:
            pass
        else:
            self.grab_data()
        self.find()


def analyze(data, fname=None):
    obj = FindStars(data, fname)
    obj.run_analysis()
    return obj.sources


def sequential(data_to_process, flist=None):
    sources = []
    total = 0
    if not flist:
        # Analyze with class and photutils dataset
        print('Using photutils data')
        start_time = time.time()
        for dset in data_to_process:
            sources.append(analyze(dset, None))
        end_time = time.time()
        total = (end_time - start_time)/60
    else:
        # Analyze with class and read data from fits file
        print('Using data from fits file')
        start_time = time.time()
        for dset in flist:
            sources.append(analyze(None, dset))
        end_time = time.time()
        total = (end_time - start_time)/60
    return total, sources


def parallel(data_to_process, use_thread, flist = None):
    sources = []
    total = 0
    if not flist:
        print('Using photutils data')
        start_time = time.time()
        delayed_objects = [dask.delayed(analyze)(dset, None) for dset in data_to_process]
        if use_thread:
            sources = dask.compute(*delayed_objects, scheduler='threads')
        else:
            sources = dask.compute(*delayed_objects, scheduler='processes')
        end_time = time.time()
        total = (end_time - start_time)/60
    else:
        print('Using data from fits file')
        start_time = time.time()
        delayed_objects = [dask.delayed(analyze)(None, dset) for dset in flist]
        if use_thread:
            print('Using threads')
            sources = dask.compute(*delayed_objects, scheduler='threads')
        else:
            print('Using processes')
            sources = dask.compute(*delayed_objects, scheduler='processes')
        end_time = time.time()
        total = (end_time - start_time)/60
    print('Total number of sources {}'.format(sum([len(val) for val in sources])))
    return total, sources

def main(parallelize=False, N=10, fname = None, use_thread = False):
    print('Running task on {} datasets'.format(N))
    hdu = datasets.load_star_image()
    flist = None
    if fname:
        flist = [fname]*N    
    data_to_process = [hdu.data]*N
    sources = []
    total = 0
    if not parallelize:
        print('Not Parallelized')
        total, sources = sequential(data_to_process, flist)
    else:
        print('Parallelized')
        total, sources = parallel(data_to_process, use_thread, flist)
    print('Processing Time: {:.4f} minutes'.format(total))
    return total, sources


if __name__ == '__main__':
    args = parser.parse_args()
    # main(args.p, args.n, args.f, args.t)
    main(True, 20, 'jd5702091_crj.fits', False)