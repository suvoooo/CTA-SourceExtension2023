# CTA-SourceExtension2023

## Neural Net Architecture (Simplified)

![Neural-Net](https://github.com/suvoooo/CTA-SourceExtension2023/blob/main/plots/plot_neural_net_ICRC2023.png-1.png)

----------------------------------------------


### Activation Maps from Conv4 Layer for Sources with Different Extensions

#### C0: $0.03 < \sigma < 0.1$

![Activation Maps](https://github.com/suvoooo/CTA-SourceExtension2023/blob/main/plots/check_conv_layers_C0_999_final.png)

#### C0: $0.1 < \sigma < 0.3$

![ActivationsMapsC1](https://github.com/suvoooo/CTA-SourceExtension2023/blob/main/plots/check_conv_layers_C1_999_final.png)

--------------------------------------------

### Libraries & Versions: 


1. Python: `3.9.12`
2. Matplotlib: `3.5.1`
3. Numpy: `1.22.3`
4. Scipy: `1.8.1`
5. Sklearn: `1.0.2`
6. TensorFlow: `2.4.1`
7. Gammapy: `1.0.1`
8. ctools: `1.7.4`

----------------------------------------------------


### scripts:  

Python scripts used for the production 

1. _**fits_to_npy_CTA_extent_sel.py:**_ Convert the fits files from ctools simulation to numpy arrays; For every fits file we have 4 numpy arrays for 4 energy bins

2. _**classification_DataLoader_CTA_Ext.py:**_ Dataloader module used for preprocessing the numpy arrays to make them suitable for our network. Also includes augmentation for the training set.

3. _**neural_nets.py:**_ Module to import the network used for our task. 

----------------------------------------------------

### notebooks:

Helping Notebooks to visualize several sections of scripts 

1. _**read_fits_augment_npy.ipynb:**_ visualize the numpy arrays (source images) after processing the fits files; Also a visualizer for the augmentations used.

2. _**Neural-Net-Arch.ipynb:**_ import neural_nets.py and check in detail the network used for analysis.  
 
