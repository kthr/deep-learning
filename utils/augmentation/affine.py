from scipy.ndimage.interpolation import rotate
import numpy as np

def apply_rotation(image, rotations, order=3):
    xy, xz, yz = rotations
    if len(np.squeeze(image).shape) == 2:
        rotated=rotate(image,xy,reshape=False, order=order)
    else:
        rotated=rotate(image,xy, axes=(1,2),reshape=False, order=order)
        rotated=rotate(rotated,xz, axes=(1,0),reshape=False, order=order)
        rotated=rotate(rotated,yz, axes=(2,0),reshape=False, order=order)
    return rotated

def random_rotation(image, max_rotations, order=3):
    xy,xz,yz=(np.random.random(3)*np.array(max_rotations)).astype("uint32")
    return apply_rotation(image, xy, xz, yz, order)

def sample_rotation(max_rotations=(90,90,90)):
    rotations=tuple((np.random.random(3)*np.array(max_rotations)).astype("uint32"))
    return rotations
