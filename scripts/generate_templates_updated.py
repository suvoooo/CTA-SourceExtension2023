#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:46:37 2022

@author: veronikavodeb
"""

import xml.etree.ElementTree as ET
import numpy as np
import ctools
import cscripts
import os
import math
from numpy import random
from scipy.stats import loguniform
import matplotlib.pyplot as plt
from astropy.io import fits

# Set the path to the models that are to be used for the interstellar and instrumental backgrounds
models_path = '/'

caldb = 'prod5-v0.1'

# | DEFINE GENERAL PARAMETERS | #

# | The energy range we will consider. |
emin       = 0.010  # TeV
emax       = 100.0  # TeV

patch_size = [1.4, 1.4] # deg

# | The energy binning should be at least 15 bins per decade. |
enumbins   = 60

binsz      = 0.02 # deg/bin

nxpix      = round(patch_size[0] / binsz)
nypix      = round(patch_size[1] / binsz)

# | Deciding on the backgrounds and physical models we will consider, with corresponding maps files. |
backgrounds = [['CR', models_path + 'bkg_irf.xml']]

# | DEFINE FUNCTIONS FOR SIMULATING EVENTS | #

def create_obs_xml_file(pointing_file, out_file_xml):
    obsdef = cscripts.csobsdef()
    obsdef['inpnt']  = pointing_file
    obsdef['outobs'] = out_file_xml
    obsdef.run()
    obsdef.save()
    print(obsdef.obs())
    return obsdef.obs()

def create_obs_xml_file_for_region(in_file_xml, out_file_xml, xref, yref, x_width, y_height):    
    obsselect = cscripts.csobsselect()
    obsselect['inobs']     = in_file_xml
    obsselect['pntselect'] = 'BOX'
    obsselect['coordsys']  = 'GAL'
    obsselect['glon']      = xref
    obsselect['glat']      = yref
    obsselect['width']     = x_width
    obsselect['height']    = y_height
    obsselect['tmin']      = 'NONE'
    obsselect['tmax']      = 'NONE'
    obsselect['outobs']    = out_file_xml
    obsselect.execute()
    return obsselect.obs()

def simulate_counts(name, in_xml_file, model, name_ext, xref, yref, proj='CAR', coordsys='GAL', 
                    binsz=binsz, nxpix=nxpix, nypix=nypix, emin=emin, emax=emax, ebinalg='LOG', enumbins=enumbins):
    evfile = export_data_path + 'map_' + name + name_ext + '.fits'
    model_string = model
    model = ctools.ctmodel()
    model['inobs']     = in_xml_file
    model['inmodel']   = model_string
    model['outcube']   = evfile
    model['xref']      = xref
    model['yref']      = yref
    model['proj']      = proj
    model['coordsys']  = coordsys
    model['binsz']     = binsz
    model['nxpix']     = nxpix
    model['nypix']     = nypix
    model['ebinalg']   = ebinalg
    model['emin']      = emin
    model['emax']      = emax
    model['enumbins']  = enumbins
    model['incube']    = 'NONE'
    model.execute()
    return evfile

images = []

SIZE = 1000
start_id = 0000

for i in range(SIZE):
    
    # | The power-law spectrum for the source. |
    
    region_l  = round(random.uniform(low=1.0/2, high=19.0/2, size=None), 2)*2
    region_b  = round(random.uniform(low=-3.0/2, high=3.0/2, size=None), 2)*2

    dict_PL = {'l': region_l, 'b': region_b, 'withPS': True, 'spectrum': 'PL'}
    k0 = np.power(10, random.normal(loc=-12, scale=0.8, size=None))*1e-6 # /TeV/cm2/s = 1e-6 /MeV/cm2/s
    E0 = 1.0e6 # MeV
    gamma = random.normal(loc=-2.2, scale=0.1, size=None)
    a, b = 0.03, 1.0
    sigma = round(loguniform.rvs(a, b, size=None), 3)
    
    dict_PL['k0'] = k0
    dict_PL['E0'] = E0
    dict_PL['gamma'] = gamma
    dict_PL['sigma'] = sigma
    
    images.append(dict_PL)

# | GENERATING THE MAPS |

for nr, image in enumerate(images, start=start_id):
    print('Image {:04d}, sigma = {} deg'.format(nr, image['sigma']))
    name_ext_com = 'Image{:04d}_extended_'.format(nr, emin, emax)
    
    xml_fn = 'obs.xml'
    
    xref = image['l']
    yref = image['b']
    sig  = image['sigma']

    if OBS == 'GPS':
        region_ext    = 5.0
        obsdef_region = create_obs_xml_file_for_region('gps_obs.xml', xml_fn, xref, yref, patch_size[0]+region_ext, patch_size[1]+region_ext)

        name_ext_map = 'GPS_binsz{}_ebins{}_{}x{}_region{}x{}'.format(binsz, enumbins, patch_size[0], patch_size[1], 
                                                                   patch_size[0]+region_ext, patch_size[1]+region_ext)

        name_ext = '_' + name_ext_com + name_ext_map
        ln = np.linspace(xref-patch_size[0]/2, xref+patch_size[0]/2, round(patch_size[0]/binsz)+1)
        lt = np.linspace(yref-patch_size[1]/2, yref+patch_size[1]/2, round(patch_size[1]/binsz)+1)
        Lon,Lat = np.meshgrid(ln,lt)
        
        x1 = round((xref-patch_size[1]/2)/binsz)
        x2 = round((xref+patch_size[1]/2)/binsz)
        y1 = round((yref-patch_size[0]/2+4)/binsz)
        y2 = round((yref+patch_size[0]/2+4)/binsz)
        print('x1 = {}, x2 = {}, y1 = {}, y2 = {}'.format(x1,x2,y1,y2))
        
        name = ''
        hdul2d = np.zeros((round(patch_size[0] / binsz), round(patch_size[1] / binsz)))
        hdul = np.zeros((enumbins, round(patch_size[0] / binsz), round(patch_size[1] / binsz)))
        
        b = 0
        if image['withPS'] == True:
            source_name = image['spectrum']
            filename = directory_path + 'xml_files/extended_source_{}.xml'.format(image['spectrum'])
            tree = ET.parse(filename)
            root = tree.getroot()
            label = root.find(".//parameter[@name='GLON']")
            label.attrib['value'] = '%s'%xref
            label = root.find(".//parameter[@name='GLAT']")
            label.attrib['value'] = '%s'%yref
            label = root.find(".//parameter[@name='Sigma']")
            label.attrib['value'] = '%s'%sig
            label = root.find(".//parameter[@name='Prefactor']")
            label.attrib['value'] = '%s'%image['k0']
            label.attrib['min'] = '%s'%(image['k0']-1e-1)
            label.attrib['max'] = '%s'%(image['k0']+1e1)
            label.attrib['scale'] = '%s'%1.0
            label = root.find(".//parameter[@name='PivotEnergy']")
            label.attrib['value'] = '%s'%image['E0']
            label.attrib['min'] = '%s'%(image['E0']-1e-1)
            label.attrib['max'] = '%s'%(image['E0']+1e1)
            label.attrib['scale'] = '%s'%1.0
            label = root.find(".//parameter[@name='Index']")
            label.attrib['value'] = '%s'%image['gamma']
            label.attrib['min'] = '%s'%(image['gamma']-1e-1)
            label.attrib['max'] = '%s'%(image['gamma']+1e1)
            label.attrib['scale'] = '%s'%1.0
                
            if image['spectrum'] == 'LP':
                label = root.find(".//parameter[@name='Curvature']")
                label.attrib['value'] = '%s'%image['eta']
                label.attrib['min'] = '%s'%(image['eta']-1e-1)
                label.attrib['max'] = '%s'%(image['eta']+1e1)
                label.attrib['scale'] = '%s'%1.0
                
            elif image['spectrum'] == 'EC':
                label = root.find(".//parameter[@name='CutoffEnergy']")
                label.attrib['value'] = '%s'%image['Ecut']
                label.attrib['min'] = '%s'%(image['Ecut']-1e-1)
                label.attrib['max'] = '%s'%(image['Ecut']+1e1)
                label.attrib['scale'] = '%s'%1e6
            
            tree.write(filename, encoding='UTF-8', xml_declaration=True)

            name = 'source-{}+'.format(image['spectrum'])
            file = simulate_counts('GS', 'obs.xml', filename, name_ext, xref, yref)
            source_hdul = fits.open(file)[0].data
            source_hdul2d = source_hdul.sum(axis=0)
            b = 1
            
        # Define the instrumental background map that is used as input for the background model.
        file = 'map_CR.fits'

        hdul = np.flip(fits.open(file)[0].data, axis=(1,2))
        region = hdul[:, range(y1,y2)]
        region = region[:,:,range(x1, x2)]

        REGION = np.add(region, source_hdul)
        name = name + 'CR+'

        split_at = [7,15,30,45]
        def split_array(data, split_at=split_at):
            split = np.split(data, [7,15,30,45], axis=0)
            return [np.sum(split[i], axis=0) for i in range(1,len(split))]

        hdu1 = fits.PrimaryHDU(split_array(source_hdul))
        hdu2 = fits.ImageHDU(split_array(region))
        hdu3 = fits.ImageHDU(split_array(REGION))
        hdu4 = fits.ImageHDU(source_hdul)
        hdu5 = fits.ImageHDU(region)
        hdu6 = fits.ImageHDU(REGION)
        hd = fits.HDUList([hdu1])

        hd[0].header.append(('IMGCONT', 'SOURCE', 'The content of this image'), end=True)
        hd[0].header.append(('CRPIX1', round(patch_size[0] / binsz / 2), 'Pixel value of reference point'), end=True)
        hd[0].header.append(('CRPIX2', round(patch_size[1] / binsz / 2), 'Pixel value of reference point'), end=True)
        hd[0].header.append(('XREF', xref, '[deg] Longitude coordinate of reference point'), end=True)
        hd[0].header.append(('YREF', yref, '[deg] Latitude coordinate of reference point'), end=True)
        hd[0].header.append(('SIGMA', image['sigma'], '[deg] Gaussian extension of the spatial shape'), end=True)
        hd[0].header.append(('PREFAC', image['k0'], '[phcm-2s-1MeV-1] Prefactor parameter value for the spectrum'), end=True)
        hd[0].header.append(('PIVEN', image['E0'], '[MeV] Pivot energy parameter value for the spectrum'), end=True)
        hd[0].header.append(('INDEX', image['gamma'], '[ ] Index parameter value for the spectrum'), end=True)

        fits_file = export_data_path + 'Images_ExtendedSources/' + name_ext_com + name + '.fits'
        hd.writeto(fits_file, overwrite=True)

        hdr = fits.getheader(file, 1)

        hdr['EXTNAME'] = 'CRBKG'
        hdr.append(('XREF', xref, '[deg] Longitude coordinate of reference point'), end=True)
        hdr.append(('YREF', yref, '[deg] Latitude coordinate of reference point'), end=True)
        fits.append(fits_file, split_array(region), hdr)

        hdr['EXTNAME'] = 'SRC+CR'
        hdr.append(('SIGMA', image['sigma'], '[deg] Gaussian extension of the spatial shape'), end=True)
        hdr.append(('PREFAC', image['k0'], '[phcm-2s-1MeV-1] Prefactor parameter value for the spectrum'), end=True)
        hdr.append(('PIVEN', image['E0'], '[MeV] Pivot energy parameter value for the spectrum'), end=True)
        hdr.append(('INDEX', image['gamma'], '[ ] Index parameter value for the spectrum'), end=True)
        fits.append(fits_file, split_array(REGION), hdr)

        hdr['EXTNAME'] = 'SOURCE'
        fits.append(fits_file, source_hdul, hdr)

        hdr['EXTNAME'] = 'CRBKG'
        del hdr['SIGMA']
        del hdr['PREFAC']
        del hdr['PIVEN']
        del hdr['INDEX']
        fits.append(fits_file, region, hdr)

        hdr['EXTNAME'] = 'SRC+CR'
        hdr.append(('SIGMA', image['sigma'], '[deg] Gaussian extension of the spatial shape'), end=True)
        hdr.append(('PREFAC', image['k0'], '[phcm-2s-1MeV-1] Prefactor parameter value for the spectrum'), end=True)
        hdr.append(('PIVEN', image['E0'], '[MeV] Pivot energy parameter value for the spectrum'), end=True)
        hdr.append(('INDEX', image['gamma'], '[ ] Index parameter value for the spectrum'), end=True)
        fits.append(fits_file, REGION, hdr)
        
