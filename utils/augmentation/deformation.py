import numpy as np
from skimage.filters import gaussian
from scipy.interpolate import RegularGridInterpolator

def random_deformation_field(shape=(100,100,100), max_displacement=20, sigma=3):
    field=gaussian((np.random.random(shape+(len(shape),))*(max_displacement)-max_displacement/2)*2, sigma=sigma, multichannel=True)
    return field

def apply_deformation_field(image, field, method="linear"):
    shape=field.shape[0:-1]
    ranges=tuple([range(0,s) for s in shape])
    f=RegularGridInterpolator(ranges, image,bounds_error=False, fill_value=0)
    coordinates=np.meshgrid(*ranges, indexing="ij")
    if len(shape)==2:
        displacement=np.transpose(coordinates,(1,2,0))+field
    else:
        displacement=np.transpose(coordinates,(1,2,3,0))+field
    displacement=np.reshape(displacement, (np.prod(shape),displacement.shape[-1]))
    deformed=f(displacement)
    return np.reshape(deformed, image.shape)

def random_deformation(image, max_displacement=20, sigma=3, method="linear"):
    field=random_deformation_field(np.squeeze(image).shape,max_displacement, sigma)
    return apply_deformation_field(image,field, method) 

def checkboard3D(ncheckers_pairs=5, scale=10):
    return np.kron([[[1,0] * ncheckers_pairs, [0,1]*ncheckers_pairs] *ncheckers_pairs, [[0,1] * ncheckers_pairs, [1,0]*ncheckers_pairs] *ncheckers_pairs]*ncheckers_pairs, np.ones((scale,scale,scale)))

def checkboard2D(ncheckers_pairs=5, scale=10):
    return np.kron([[1,0] * ncheckers_pairs, [0,1]*ncheckers_pairs] *ncheckers_pairs, np.ones((scale,scale)))