import glob, random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import poisson
from skimage.transform import rotate



###############################################
# fits_to_npy_CTA_extent_sel.py: turns fits to npy and here we deal with npy arrays
#
# Helper Functions for reading and augmenting data
# 
# Augmenting data [set different augmentation options]
# used: flip ud, flip lr, rotate 90 (counter clock)
###############################################

   

def norm_arr(arr1): # not used
   '''
   standardization: subtract mean 
   '''
   return (arr1 - np.mean(arr1) + 1e-9) /(np.std(arr1) + 1e-9)

def norm_max_arr(arr): # used
   '''
   standardization: everything between 0 and 1
   '''
   return (arr /(np.max(arr) + 1e-9)) 

def norm_max_arr_poisson1(arr): # not used: images already include poisson statistics
   return poisson.rvs(norm_max_arr(arr))
 


###############################################
# classes based on extensions
# C0: 0.03-0.1
# C1: 0.1-0.3
# C2: 0.3-0.99
###############################################
class0, class1, class2  = 'C0', 'C1', 'C2'
###############################################

###############################################
# energy bins: 4
# E0: 30 GeV - 100 GeV
# E1: 100 GeV - 1 TeV
# E2: 1 TeV - 10 TeV
# E3: 10 TeV - 100 TeV
############################################### 



filepath0_ext = '/path/to/npy/images/%s/' % (class0)
filepath1_ext = '/path/to/npy/images/%s/' % (class1)
filepath2_ext = '/path/to/npy/images/%s/' % (class2)

csv_filepath = '/path/to/npy/images/'






def read_train_source_ext(cls_label:int, bin:int, less_data=False):
    '''
    spider through the folders and load images as arrays
    
    preparation for array for a particular energy bin and particular class  
    '''

    ##############################################
    # filepaths
    ##############################################
    print ('allowed values for class labels: 0, 1, 2')



    if cls_label==0:
       filepath = filepath0_ext
       # all_files_list = glob.glob(filepath0 + '/*.npy')
    elif cls_label==1:
       filepath = filepath1_ext
       # all_files_list = glob.glob(filepath1 + '/*.npy')
    elif cls_label==2:
       filepath = filepath2_ext
       # all_files_list = glob.glob(filepath2 + '/*.npy')
    else:
       print ('allwoed values for class labels should be 0, 1, or 2')


    all_ims = []
    all_fnames = []
    # all_labels_a = []

    if bin==0:
        filepath = filepath + 'E0/'
        all_files_list = glob.glob(filepath + '*.npy')
    elif bin==1:
        filepath = filepath + 'E1/'
        all_files_list = glob.glob(filepath + '*.npy')
    elif bin==2:
        filepath = filepath + 'E2/'
        all_files_list = glob.glob(filepath + '*.npy')
    elif bin==3:
        filepath = filepath + 'E3/'
        all_files_list = glob.glob(filepath + '*.npy')        
    else:
        print ('allowed values for energy bins are: 0, 1, 2, 3', )    
      

    if less_data:
       all_files_list_sel = random.sample(all_files_list, int(len(all_files_list)*0.90)) 
       #select randomly as much data as you need
       for f in all_files_list_sel:
        im_load = np.load(f)
        all_fnames.append(f)   
        im_norm = norm_max_arr(im_load)# saved images were already poisson
        im_norm_re = np.expand_dims(im_norm, axis=-1)           
        all_ims.append(im_norm_re)
    else:
       all_files_list_sel = all_files_list   
       for f in all_files_list_sel:
        im_load = np.load(f)
        all_fnames.append(f)   
        im_norm = norm_max_arr(im_load)
        im_norm_re = np.expand_dims(im_norm, axis=-1)           
        all_ims.append(im_norm_re)
    return np.array(all_ims), np.array(all_fnames)







############################################################################
# train data augmentation
############################################################################


def aug_ims_source_CTA(all_ims, rotate_90=True, rotate_180=True, augment=True):
   print ('performs generic augmentation: Flip UD, Flip LR, Rotate 90, 180')
   
   print ('Flip UD and Flip LR always happen until augment is False')

   ims_allE_aug = []
   
   ims_allE_= []
   
   ims_allE_aug_90 = []

   ims_allE_aug_lr = []

   ims_allE_aug_180 = []


   for j in range(len(all_ims)):
      source_im = all_ims[j]
      source_im_ud = np.flipud(source_im)
      source_im_lr = np.fliplr(source_im)
      source_im_90 = np.rot90(source_im, k=1, axes=(0, 1))
      source_im_180 = rotate(source_im, 180) # never used
      ims_allE_.append(source_im)
      ims_allE_aug.append(source_im_ud)
      ims_allE_aug_90.append(source_im_90)
      ims_allE_aug_180.append(source_im_180)
      ims_allE_aug_lr.append(source_im_lr)
      if augment==False:
        all_ims_aug_t0 = ims_allE_
      elif rotate_90 & augment & (rotate_180==False):
        all_ims_aug_t0 = ims_allE_ + ims_allE_aug + ims_allE_aug_lr + ims_allE_aug_90
      elif rotate_180 & augment & (rotate_90==False):   
        all_ims_aug_t0 = ims_allE_ + ims_allE_aug + ims_allE_aug_lr + ims_allE_aug_180
      elif rotate_90 & rotate_180 & augment:
        all_ims_aug_t0 = ims_allE_ + ims_allE_aug + ims_allE_aug_lr + ims_allE_aug_180 + ims_allE_aug_90
      elif rotate_90 & rotate_180 & (augment==False):
        all_ims_aug_t0 = ims_allE_    
      elif (rotate_90 == False) & (rotate_180 == False) & augment:
        all_ims_aug_t0 = ims_allE_ + ims_allE_aug + ims_allE_aug_lr   
      

      all_source_im_aug_t0_arr = np.array(all_ims_aug_t0)
      
   final_source_allE_allT_aug = np.array(all_source_im_aug_t0_arr) 
   
   ims_allE_.clear()
   ims_allE_aug.clear()
   ims_allE_aug_90.clear()
   ims_allE_aug_180.clear()
   ims_allE_aug_lr.clear()   

   return final_source_allE_allT_aug


# ####################################
# # Test data are not augmented
# ####################################
