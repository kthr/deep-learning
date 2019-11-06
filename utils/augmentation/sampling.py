import numpy as np
from joblib import Parallel, delayed

from .affine import *
from .deformation import *

def select_image_samples(image, shape=(64,64,64), n=10, seed=None, with_augmentation=False):
    """
    Select n samples from an image (z,x,y) with the given shape. Returns the sampled positions as (x,y,z) coordinates in image space.

    Args:
        image (numpy.array): an 3D image or array of images
        shape (tuple): the sample shape
        n (int): the number of samples to be sampled
        seed (int): a random seed for reproducability of the sampling

    Returns:
        (np.array) returns an array of sampled positions
    """
    positions=[]
    if seed!=None:
        np.random.seed(seed)
    if with_augmentation:
        sp=np.power(shape,2)
        diagonal=np.ceil(np.sqrt(np.max([sp[0]+sp[1],sp[1]+sp[2],sp[2]+sp[0]]))).astype("uint32")
        padding=np.ceil(np.max(np.array(diagonal-shape))).astype("uint32")
    else:
        padding=0
    new_shape=np.array(shape)+2*padding
    while len(positions) < n:
        z = np.random.randint(padding,image.shape[0]-new_shape[0]-1)
        x = np.random.randint(padding,image.shape[1]-new_shape[1]-1)
        y = np.random.randint(padding,image.shape[2]-new_shape[2]-1)  
        positions.append([z,x,y])
    return positions

def get_image_samples(image, positions, shape=(64,64,64), with_rotation=False, max_rotations=(90,90,90), order=3, with_deformation=False, max_displacement=20, sigma=3, seed=None, cores=1):
    """
    Extract samples of the given shape from an image (z,x,y) at the given positions.

    Args:
        image (numpy.array): an 3D image or array of images
        shape (tuple): the sample shape
        positions (np.array): the sample positions
    Returns:
        (np.array) returns an array of sampled sub-images
    """
    if order>0:
        method="linear"
    else:
        method="nearest"
    if seed != None:
        np.random.seed(seed)
        
    if with_rotation or with_deformation:
        sp=np.power(shape,2)
        diagonal=np.ceil(np.sqrt(np.max([sp[0]+sp[1],sp[1]+sp[2],sp[2]+sp[0]]))).astype("uint32")
        padding=np.ceil(np.max(np.array(diagonal-shape))).astype("uint32")
    else:
        padding=0    
    if with_deformation:
        sample_shape=_crop_sample(image, positions[0], padding, shape).shape
        deformations=np.array([random_deformation_field(sample_shape, max_displacement, sigma) for i in range(len(positions))])
    else:
        deformations=np.repeat(None,len(positions))
    if with_rotation:
        rotations=np.array([sample_rotation(max_rotations) for i in range(len(positions))])
    else:
        rotations=np.repeat(None,len(positions))
    
    samples=np.array(Parallel(cores)(delayed(_crop_sample)(image,pos,padding,shape) for pos in positions))
    samples=np.array(Parallel(cores)(delayed(_create_sample)(sample, shape, padding, rot, deform, order) for sample,rot,deform in zip(samples,rotations,deformations)))
    if image.shape[-1] == 1:
        return np.expand_dims(samples, axis=-1)    
    else:
        return samples

def _create_sample(sample, shape, padding=0, rotations=None, deformation=None, order=3):
    if type(rotations) != type(None):
        sample=apply_rotation(sample, rotations, order)
    if type(deformation) != type(None):
        method = "linear" if order>0 else "nearest"
        sample=apply_deformation_field(sample, deformation, method)
    return np.squeeze(sample[padding:padding+shape[0],padding:padding+shape[1],padding:padding+shape[2]])

def _crop_sample(image, position, padding, shape):
    z,x,y=position
    return image[(z-padding):z+shape[0]+padding,(x-padding):x+shape[1]+padding,(y-padding):y+shape[2]+padding]