#############################
# this program reads the fits files
# selects the src + cr component
# seperate 4 bins
# add corresponding poisson statistics
# based on the source extension we add different keywords ('C0', 'C1', 'C2')
# save the arrays as numpy arrays 
#############################

import glob

import numpy as np 
#import pandas as pd
from astropy.io import fits 

import matplotlib.pyplot as plt

from scipy.stats import poisson

filepath = '/path/to/file/'

save_filepath = '/path/to/dir/'

# 4 bins [30 GeV, 100 GeV], [100 GeV, 1TeV], [1 TeV, 10TeV], [10 TeV, 100 TeV] 


######################################
# read fits and separate 4 bins
######################################
def gen_func_save(filename):
    '''
    input: fits file generated via ctools
    output: src + CR at 4 different bins, corresponding filename
    '''
    fits_file = fits.open(filename)
    image_data = fits_file[2].data # SRC + CR (4, 70, 70)
    image_data = np.moveaxis(image_data, 0, -1)
    image_data_bin_0 = image_data[:, :, 0]
    image_data_bin_1 = image_data[:, :, 1]
    image_data_bin_2 = image_data[:, :, 2]
    image_data_bin_3 = image_data[:, :, 3]
    poisson_image_0 = poisson.rvs(image_data_bin_0)
    poisson_image_1 = poisson.rvs(image_data_bin_1)
    poisson_image_2 = poisson.rvs(image_data_bin_2)
    poisson_image_3 = poisson.rvs(image_data_bin_3)

    filename_strip = filename.split('/')[-1].replace('fits', 'npy')
    
    return poisson_image_0, poisson_image_1, poisson_image_2, poisson_image_3, filename_strip  # return with added poisson noise & filenames



all_files_list = glob.glob(filepath + '/*.fits')

all_bkg_ims = []
all_source_ims = []
cls_label = []
all_sigmas = []

def check_sigmas(file_list):
    for i in range(len(file_list)):
        # if 'source' in file_list[i]:
        fits_file = fits.open(file_list[i])
        sigma = fits_file[0].header['SIGMA']
        all_sigmas.append(sigma)
    return all_sigmas    

def save_poi_im_files(file_list , save_dir):
    for i in range(len(file_list)):
        fits_file = fits.open(file_list[i])
        sigma = fits_file[0].header['SIGMA']
        if sigma >=0.03 and sigma <=0.1:
            poi_im0, poi_im1, poi_im2, poi_im3, f_name = gen_func_save(file_list[i])
            print('poi im shape:', poi_im0.shape, poi_im1.shape)
            print ('check_filename: ', f_name)
            f_name_new0 = f_name.replace('+', '_').replace('source-', 'C0_E0_')
            f_name_new1 = f_name.replace('+', '_').replace('source-', 'C0_E1_')
            f_name_new2 = f_name.replace('+', '_').replace('source-', 'C0_E2_')
            f_name_new3 = f_name.replace('+', '_').replace('source-', 'C0_E3_')
            np.save(save_dir + f_name_new0, poi_im0)
            np.save(save_dir + f_name_new1, poi_im1)
            np.save(save_dir + f_name_new2, poi_im2)
            np.save(save_dir + f_name_new3, poi_im3)
        if sigma >0.1 and sigma <=0.3:
            poi_im0, poi_im1, poi_im2, poi_im3, f_name = gen_func_save(file_list[i])
            print('poi im shape:', poi_im0.shape, poi_im1.shape)
            print ('check_filename: ', f_name)
            f_name_new0 = f_name.replace('+', '_').replace('source-', 'C1_E0_')
            f_name_new1 = f_name.replace('+', '_').replace('source-', 'C1_E1_')
            f_name_new2 = f_name.replace('+', '_').replace('source-', 'C1_E2_')
            f_name_new3 = f_name.replace('+', '_').replace('source-', 'C1_E3_')
            np.save(save_dir + f_name_new0, poi_im0)
            np.save(save_dir + f_name_new1, poi_im1)
            np.save(save_dir + f_name_new2, poi_im2)
            np.save(save_dir + f_name_new3, poi_im3)
        if sigma >0.3 and sigma <=0.999:
            poi_im0, poi_im1, poi_im2, poi_im3, f_name = gen_func_save(file_list[i])
            print('poi im shape:', poi_im0.shape, poi_im1.shape)
            print ('check_filename: ', f_name)
            f_name_new0 = f_name.replace('+', '_').replace('source-', 'C2_E0_')
            f_name_new1 = f_name.replace('+', '_').replace('source-', 'C2_E1_')
            f_name_new2 = f_name.replace('+', '_').replace('source-', 'C2_E2_')
            f_name_new3 = f_name.replace('+', '_').replace('source-', 'C2_E3_')
            np.save(save_dir + f_name_new0, poi_im0)
            np.save(save_dir + f_name_new1, poi_im1)
            np.save(save_dir + f_name_new2, poi_im2)
            np.save(save_dir + f_name_new3, poi_im3)        
    return True   

        
save_poi_im_files(all_files_list, save_filepath)

all_sigmas = check_sigmas(all_files_list)

print ('max, min, med sigma: ', max(all_sigmas), min(all_sigmas), np.median(all_sigmas))

hist, edges = np.histogram(all_sigmas, bins=[0.028, 0.1, 0.3, 0.999]) # 
print (hist, '\n', edges)
