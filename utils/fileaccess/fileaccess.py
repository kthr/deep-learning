#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from joblib import Parallel, delayed
from skimage.io import imread, imsave
import numpy as np
import os
import re
import sys
import warnings

from .decorators import *

def file_names(pattern, directory, level=1):
    """
    Get all file names matching the regex pattern.

    Args:
        pattern (str): string defining a regex pattern
        directory (str): path of the directory to search
        level (int): the highest level of the directory tree that will be searched

    Returns:
        (list, str) names of all the files that have been found
    """
    return _names(pattern, directory, level, False)

def directory_names(pattern, directory, level=1):
    """
    Get all directory names matching the regex pattern.

    Args:
        pattern (str): string defining a regex pattern
        directory (str): path of the directory to search
        level (int): the highest level of the directory tree that will be searched

    Returns:
        (list, str) names of all the directories that have been found
    """
    return _names(pattern, directory, level, True)



def import_image(filename, scale=True, expand=None, statistics=True):
    """
    Read an image file.

    Args:
        filename (str): a file name (path)
        scale (bool): should the returned image be scaled to float32 [0,1]
        expand (int): expands the specified axis of the image (for keras with channels last this should be -1)
        verbose (bool): print information about the image

    Returns:
        (numpy.array) returns an numpy array containing the image
    """
    if(filename.endswith('.tif')):
        image=imread(filename, plugin='tifffile')
    else:
        image=imread(filename, plugin='simpleitk')
    if scale:
        if image.dtype==np.bool:
            image=image.astype('float32')
        elif image.dtype==np.int8:
            image=(image.astype('float32')+(2**8/2))/(2**8-1)
        elif image.dtype==np.int16:
            image=(image.astype('float32')+(2**16/2))/(2**16-1)
        elif image.dtype==np.uint8:
            image=image.astype('float32')/(2**8-1)
        elif image.dtype==np.uint16:
            image=image.astype('float32')/(2**16-1)
        else:
            warnings.warn('Scaling for dtype {} is not yet implemented!'.format(image.dtype))
    if expand!=None:
        image=np.expand_dims(image,axis=expand)
    print("Image info for: {}".format(os.path.basename(filename)))
    image_info(image, statistics)
    return image

def import_image_list(filenames, scale=True, expand=None, statistics=True, parallel_processes=4):
    """
    Reads all the images given.

    Args:
        filenames (list): a list of file names (paths)
        scale (bool): should the returned image be scaled to float32 [0,1]
        verbose (bool): print information about the image
        parallel_processes (int): the number of parallel processes used to read the files

    Returns:
        (numpy.array) returns an numpy array containing the images
    """
    images=np.array(Parallel(n_jobs=parallel_processes)(delayed(import_image)(i, scale, expand, False) for i in filenames))
    if statistics and len(images)>0:
        image_info(images)
    return images

def export_image_list(path, image_arrays, prefix='image', extension='tif', digits=None):
    """
    Saves all the images given.

    Args:
        path (str): the path to save the images to
        image_arrays (np.array): the images in a single numpy array
        prefix (str): the image prefix
        extension (str): the image file extension
        digits (uint): the number of digits appended to the prefix (with digits=2 it is prefix00.extension, prefix01.extension ...)

    Returns:
        (void) returns nothing
    """
    if digits==None:
        nd=len(str(len(image_arrays)))
    else:
        nd=int(digits)
    digit_string='{:0>'+str(nd)+'}'
    [imsave(os.path.join(path, prefix + digit_string.format(i) + '.' + extension), image_arrays[i]) for i in range(len(image_arrays))]


def image_info(image, statistics=True):
    """
    Prints some information on the given image.

    Args:
        image (np.array): the image

    Returns:
    """
    print('Image shape: {}'.format(image.shape))
    print('Image dtype: {}'.format(image.dtype))
    if statistics:
        q=np.quantile(image,[0.,.01,.99,1.])
        print('Image min, max: {}, {}'.format(q[0],q[3]))
        print("Image 1%, 99% quantiles: {}, {}".format(q[1],q[2]))

@deprecated
def parallel_imread(filenames, crop=(0,0,0,0), num_cores=12):
    """
    Read image files in parallel.

    Args:
        filenames (list): a list of file names (paths)
        num_cores (int): the number of cores used to read images

    Returns:
        (numpy.array) returns an numpy array containing all the images
    """
    return np.array(Parallel(n_jobs=num_cores)(delayed(_read_and_crop_image)(i, crop) for i in filenames))

@deprecated
def parallel_imwrite(path, image_array, prefix='image', extension='png', num_cores=12):
    """
    Writes the images in image_array to the path.

    Args:
        path (str): the path to write the images
        image_array (numpy.array): the images
        prefix (str): the filename prefix for the images
        extension (str): the image filename extension
        num_cores (int): the number of cores used for parallel writing

    Returns:
        (numpy.array) returns an numpy of all the filenames writen to the disk
    """
    num_digits=len(str(len(image_array)))
    return np.array(Parallel(n_jobs=num_cores)(delayed(_write_image)(path, image_array[i], prefix, num_digits, i, extension) for i in range(len(image_array))))

@deprecated
def _write_image(path, image, prefix, num_digits, index, extension):
    filename=os.path.join(path, prefix+'{0:0>{1:}}.'.format(index,num_digits)+extension)
    if len(image.shape)>2 and image.shape[-1]==1:
        imsave(filename, np.squeeze(image, axis=-1))
    else:
        imsave(filename, image)
    return filename

@deprecated
def _read_and_crop_image(filename, crop):
    image=imread(filename)
    left, right, top, bottom = crop
    ydim, xdim = image.shape[0:2]
    return image[top:ydim-bottom,left:xdim-right]

def _names(pattern, directory, level=1, find_directories=False):
    try:
        prog = re.compile(pattern)
    except re.error:
        print('\"{}\" is not a valid regex pattern...'.format(pattern))
        raise
    if(level > 0):
        entries = [os.path.join(directory, entry) for entry in os.listdir(directory)]
        directories = [entry for entry in entries if os.path.isdir(entry)]
        if find_directories:
            files = [entry for entry in entries if os.path.isdir(entry) and re.match(pattern,os.path.basename(entry))]
        else:
            files = [entry for entry in entries if os.path.isfile(entry) and re.match(pattern,entry)]

        [files.extend(file_names(pattern, directory_entry, level-1)) for directory_entry in directories]
        files.sort()
        return files
    else:
        return []
