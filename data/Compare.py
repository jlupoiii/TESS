import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import pickle

plt.style.use(astropy_mpl_style)


# Steps:
# look at .out file with all angles
# make dictionary with key=ffi_number and value equal to a tuple of all angles/values we want
# pickle dictionary to the angle folder
#
# look at the fits files with all of the numpy arrays for each image
# clip, median filter to reduce resolution, sigma-clip, and normalize the image
# pickle the image to the ccd folder, then repeat for each image

class Compare:
    def __init__(self, fits_folder_paths, comparisons_folder_path):
        '''
        
        '''
        self.fits_folder_paths = fits_folder_paths
        self.comparisons_folder_path = comparisons_folder_path
