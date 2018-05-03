# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:00:45 2018

@author: Binkowsky
"""

import os

imdir="images" # We will creat a folder named "images" which will contain all the taining images
if not os.path.isdir(imdir):
    os.mkdir(imdir) # creat this image dirctory if it is not exist

matt_damon=[folder for folder in os.listdir('.') if 'matt' in folder]
#looking the current working dirctory, combine all the folder which has "matt" in its name

i = 0
for folder in matt_damon:
    for imfile in os.scandir(folder):
        os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(i))) 
        i += 1 # rename all the images from 0 to 9999
