#!/usr/bin/env python

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval

import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from photutils import CircularAperture

class FindStars(object):
    """docstring for FindStars"""
    def __init__(self, data):
        super(FindStars, self).__init__()
        self.data = data
        self.sources = None
        self.fname = None

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

def main():
    pass


if __name__ == '__main__':
    main()