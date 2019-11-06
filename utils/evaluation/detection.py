import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.ndimage.measurements import center_of_mass, label
from scipy.sparse import coo_matrix
import warnings

def object_centers_from_binary(binary,structure=None, as_matrix=False):
    """
    Extracts the objects centers from a binary image.

    Args:
        binary (numpy.array): an image array
        structure (numpy.array): the neighboorhood structure coded as an numpy array of ones and zeros, if None a neighborhood with corner neighbors is automatically chosen
        as_matrix (bool): return matrix or centers
        

    Returns:
        (numpy.array) returns a list of coordinates or a matrix with the same dimensions as binary with the object center pixels set to 1
    """
    if binary.dtype != np.bool:
        raise ValueError("Image 'binary' has dtype '{}', dtype 'bool' is expected!".format(binary.dtype))
    if structure==None and len(np.squeeze(binary).shape)==2:
        struct=np.ones((3,3))
    elif structure==None:
        struct=np.ones((3,3,3))
    else:
        struct=structure
    labels, num_features=label(np.squeeze(binary),structure=struct)
    centers=np.array(center_of_mass(np.squeeze(binary),labels,range(1,num_features+1)))
    if as_matrix:
        matrix=points_to_matrix(centers,binary.shape)
        return(matrix)
    else:
        return(centers)

def unique_point_associations(ground_truth_points, predicted_points,max_distance=np.infty):
    """
    Computes a unique point association on the basis of the minimal distance between points. Given two points x and y, if the closest
    point to x is y and the closest point to y is x, then they are considered associated.

    Args:
        ground_truth_points (numpy.array): a list of coordinates as from "object_centers_from_binary"
        predicted_points (numpy.array): a list of coordinates as from "object_centers_from_binary"
        max_distance (float): the maximum distance for two points x and y to be considered associated

    Returns:
        (numpy.array) a binary matrix with ones for all associated points at a distance smaler than max_distance, the matrix has the shape (ground_truth_points.length, predicted_points.length)
    """
    sdm=point_to_point_distance(ground_truth_points,predicted_points)
    
    #select the minimum value per column that is smaller than max_distance and create sparse matrix from that
    row=np.array(range(0,sdm.shape[0]))
    col=np.argmin(sdm.toarray(),axis=1)
    values=np.min(sdm.toarray(),axis=1)
    data=np.ones(row.shape[0])
    selection=np.where(values<=max_distance) 
    left=coo_matrix((data[selection],(row[selection],col[selection])), shape=sdm.shape)
    
    #select the minimum value per row that is smaller than max_distance and create sparse matrix from that
    row=np.argmin(sdm.toarray(),axis=0)
    col=np.array(range(0,sdm.shape[1]))
    values=np.min(sdm.toarray(),axis=0)
    data=np.ones(row.shape[0])
    selection=np.where(values<=max_distance) 
    right=coo_matrix((data[selection],(row[selection],col[selection])), shape=sdm.shape)
    
    #calculate uniques
    uniques=left.multiply(right)
    distances=(uniques.multiply(sdm)).toarray()
    uniques=np.array([_zero_map(i) for i in uniques.toarray()])
    uniques=np.transpose(np.array([_zero_map(i) for i in np.transpose(uniques)]))
    return((distances<max_distance).astype('uint8')*uniques)

def point_to_point_distance(point_list1, point_list2, sparse=True):
    """
    Computes the point to point distance matrix for two point lists.

    Args:
        point_list1 (numpy.array): a list of points 
        point_list2 (numpy.array): a list of points
        sparse (bool): if True returns a sparse matrix
        
    Returns:
        (numpy.array or sparse matrix) a matrix with all the distance between points
    """
    tree1=KDTree(point_list1)
    tree2=KDTree(point_list2)
    sdm=tree1.sparse_distance_matrix(tree2,np.infty)
    if(sparse):
        return sdm
    else:
        return sdm.toarray()

def detection_summary_from_binary(ground_truth, prediction, max_distance=100, structure=None):
    """
    Calculates precision, recall and f_score of object associations from two binary images. The associations are calculated on the basis of the 
    center of mass of all objects in the two given bianry images, see "unique_point_associations".

    Args:
        ground_truth (numpy.array): an binary image array
        prediction (numpy.array): another binary image array
        max_distance (float): the maximum distance for two points x and y to be considered associated
        structure (numpy.array): the neighboorhood structure coded as an numpy array of ones and zeros, if None a neighborhood with corner neighbors is automatically chosen

    Returns:
        (tuple) precision, recall, f_score
    """
    gt_points=object_centers_from_binary(ground_truth,structure)
    prediction_points=object_centers_from_binary(prediction,structure)
    return detection_summary_from_points(gt_points, prediction_points, max_distance)
    
def detection_summary_from_points(gt_points, prediction_points, max_distance=100):
    """
    Calculates precision, recall and f_score of object associations from two binary images. The associations are calculated on the basis of the 
    center of mass of all objects in the two given bianry images, see "unique_point_associations".

    Args:
        gt_points (numpy.array): of points
        prediction_points (numpy.array): another array of points
        max_distance (float): the maximum distance for two points x and y to be considered associated

    Returns:
        (tuple) precision, recall, f_score
    """
    if len(gt_points)==0 or len(prediction_points)==0:
        if len(gt_points)==len(prediction_points):
            return 1,1,1
        else:
            return 0,0,0
    associations=unique_point_associations(gt_points, prediction_points, max_distance=max_distance)
    associations=(associations>0).astype('uint8')
    precision=np.sum(associations)/associations.shape[1]
    recall=np.sum(associations)/associations.shape[0]
    if precision+recall!=0.:
        f_score=2*precision*recall/(precision+recall)
    else:
        f_score=0.
    return precision, recall, f_score

def points_to_matrix(points, shape, scale=None):
    """
    Given some points this functions returns a matrix of shape 'shape' with the given points set to 1.

    Args:
        points (numpy.array): array of points
        shape (tuple): the shape of the matrix
        scale (tuple): scale the coordinates, e.g. (2,1,1) which scales z by 2 and x and y by 1
    Returns:
        (np.array) the matrix with points set to 1
    """
    if scale==None:
        scale=1
    matrix=np.zeros(shape,dtype='uint8')
    for p in (points*scale).astype('uint32'):
        matrix[tuple(p)]=1
    return(matrix)

def fiji_to_matrix(csv_file, shape, scale=(1,1,1)):
    """
    Given a csv file from the fiji 3D annotation tool, returns the corresponding image with the points set to 1.

    Args:
        csv_file (str): array of points
        shape (tuple): the original shape of the image, e.g. (256,256,256)
        scale (tuple): scale the coordinates, e.g. (2,1,1) which scales z by 2 and x and y by 1
    Returns:
        (np.array) the matrix with points set to 1
    """
    csv=pd.read_csv(csv_file)
    points=(csv[['Slice','Y','X']].values).astype('uint32')
    points=np.array([np.maximum(p,[1,1,1]) for p in points])
    locations=points_to_matrix(points-1,shape, scale)
    return locations

def _zero_map(row):
    if np.sum(row)==1:
        return row
    else:
        return 0*row