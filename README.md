# CTA-SourceExtension2023

### Libraries & Versions: 


1. Python: `3.9.12`
2. Matplotlib: `3.5.1`
3. Numpy: `1.22.3`
4. Scipy: `1.8.1`
5. Sklearn: `1.0.2`
6. TensorFlow: `2.4.1`
7. Gammapy: `1.0.1`
8. ctools: ``

----------------------------------------------------


### scripts:  

Python scripts used for the production 

1. fits_to_npy_CTA_extent_sel.py: Convert the fits files from ctools simulation to numpy arrays; For every fits files we have 4 numpy arrays for 4 energy bins
2. classification_DataLoader_CTA_Ext.py: Dataloader module used for preprocessing the numpy arrays to make them suitable for our network. Also includes augmentation for training set.
3. neural_nets.py: Module to import the network used for our task. 

----------------------------------------------------

### notebooks:

Helping Notebooks to visualize several sections of scripts 

1. read_fits_augment_npy.ipynb: visualize the numpy arrays (source images) after processing the fits files; Also a visualizer for the augmentations used.
2. Neural-Net-Arch.ipynb: import neural_nets.py and check in detail the network used for analysis.  
 
