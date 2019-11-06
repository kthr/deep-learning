import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap, BoundaryNorm

def image_grid(image_arrays, ncols=1, nrows=1, figsize=(10,10), cmap='Greys_r', norm=None):
    """
    Shows a list of 2D images in a grid like fashion. If no row or columns numbers are given, all images are shown in a single row.

    Args:
        image_arrays (list): a list of image arrays
        n_cols (int): the number of columns of the image_grid
        n_rows (int): the number of rows of the grid
        figsize (tuple): a tuble (int, int) specifying the image size
        cmap (str): a colormap
        norm: a matplotplib normalization such as BoundaryNorm or Normalize

    Returns:
        (pyplot) returns matplotlib plot
    """
    if len(image_arrays[0].shape)==2 or image_arrays[0].shape[-1] > 1: #test for grayscale image
        def f(x): return x
    else:
        def f(x): return np.squeeze(x,axis=2)
    if len(image_arrays) == 1: #only a single image given
        fig, axes = plot.subplots(nrows=1, ncols=1, figsize=figsize)
        plot.imshow(f(image_arrays[0]), cmap=cmap, norm=norm)
        axes.axis('off')
    else:
        if(ncols*nrows < len(image_arrays)):
            if ncols==1 and nrows==1:#if neither nrows or ncols are specified
                fig, axes = plot.subplots(nrows=1, ncols=len(image_arrays), figsize=figsize)
                [axes[i].imshow(f(image_arrays[i]), cmap=cmap, norm=norm) for i in range(len(image_arrays))]
                [axes[i].axis('off') for i in range(len(image_arrays))]
            else:
                if ncols==1 and nrows>1: #if nrows is specified
                    ncols=int(np.ceil(len(image_arrays)/nrows))
                    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
                elif ncols>1 and nrows==1:#if ncols is specified
                    nrows=int(np.ceil(len(image_arrays)/ncols))
                    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
                [axes[int((i)/ncols), i%ncols].imshow(f(image_arrays[i]), cmap=cmap, norm=norm) for i in range(len(image_arrays))]
                [axes[int((i)/ncols), i%ncols].axis('off') for i in range(len(image_arrays))]

        else:#if nrows and ncols are specified
            fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            [axes[int((i)/ncols), i%ncols].imshow(f(image_arrays[i]), cmap=cmap, norm=norm) for i in range(len(image_arrays))]
            [axes[int((i)/ncols), i%ncols].axis('off') for i in range(len(image_arrays))]

def colorize(image_arrays, ncols=1, nrows=1, figsize=(10,10), background=(1.,1.,1.,1.)):
    """
    Shows a list of 2D label images in a colored, grid like fashion. If no row or columns numbers are given, all images are shown in a single row.

    Args:
        image_arrays (list): a list of image arrays
        n_cols (int): the number of columns of the image_grid
        n_rows (int): the number of rows of the grid
        figsize (tuple): a tuble (int, int) specifying the image size
        background (tuple): a color tuple defining the background color (background is supposed to be < 0)

    Returns:
        (pyplot) returns matplotlib plot
    """
    norm, cmap = _qualitative_cmap(background=background)
    image_grid(image_arrays, ncols, nrows, figsize, cmap=cmap, norm=norm)

def _qualitative_cmap(background=(1.,1.,1.,1.),ncolors=1024):
    cmap = plot.cm.Paired
    cmaplist = [cmap(i%cmap.N) for i in range(ncolors)]
    cmaplist[0]=background
    norm = BoundaryNorm(boundaries=range(ncolors), ncolors=ncolors)
    cmap=ListedColormap(cmaplist, 'qualitative')
    return (norm,cmap)
