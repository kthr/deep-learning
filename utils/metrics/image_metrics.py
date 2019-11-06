from tensorflow.python.keras import backend as K
import numpy as np

def dice_coefficient(y_true, y_pred):
    """
    A statistic used for comparing the similarity of two samples. Here binary segmentations.

    Args:
        y_true (numpy.array): the true segmentation
        y_pred (numpy.array): the predicted segmentation

    Returns:
        (float) returns a number from 0. to 1. measuring the similarity y_true and y_pred
    """
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    smooth=1.0
    return (2*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def boundary_dice_loss(weight):
    def loss(y_true, y_pred):
        fgbg=tf.where(y_true < 2, y_true, tf.zeros_like(y_true)) #foreground and background pixels
        boundaries=tf.to_float(y_true==2)#boundary pixels
        smooth=1.0
        boundary_loss=K.sum(boundaries*y_pred)/(K.sum(boundaries)+smooth)
        return 1-(1-weight)*dice_coefficient(fgbg,y_pred)+weight*boundary_loss
    return loss

def Fbeta_loss(beta):
    '''F-beta score: "measures the effectiveness of retrieval with respect to a user who attaches β times as much importance to recall as precision"
        beta = 1: harmonic mean between precision and recall (F1 score)
        beta > 1: more focus on recall
        beta < 1: more focus on precision
    '''

    def loss(y_true, y_pred):
        beta2   = beta*beta
        smooth  = 1e-6

        A = K.flatten(y_pred)
        B = K.flatten(y_true)

        sumAB     = K.sum(A*B)
        precision = sumAB / (K.sum(A) + smooth)
        recall    = sumAB / (K.sum(B) + smooth)

        fb = (1.0+beta2) * ((precision * recall) / (((beta2 * precision) + recall) + smooth))

        return 1.0 - fb

    return loss

class MSSSIM_2D_Objective():
    """
    This is an Multi-Scale SSIM [2], based on version of:
    Helder C. R. de Oliveira
    https://github.com/helderc/src/blob/master/SSIM_Index.py
    
    To use it as normal SSIM with Gaussian kernel [1] just pass a single sigma
    on initialization.
    
    References:
    [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13(4):600--612, 2004.
    [2] H. Zhao, O. Galo, I. Frosio and J. Kautz.
        Loss Functions for Image Restoration With Neural Networks.
        IEEE Transactions on Computational Imaging, 3(1):47--57, 2017.
    
    Examples:    
    Use as loss function for images normalized to intensities 0.0.-1.0:
       msssim_loss = MSSSIM_2D_Objective()
       print('loss')
       print(msssim_loss(image1, image2))
       print('ms-ssim map')
       print(msssim_loss.msssim(image1, image2))
       
    Use as SSIM loss function:
       msssim_loss = MSSSIM_2D_Objective(kernel_size=(11,11), sigmas=[1.5])
       print('loss')
       print(msssim_loss(image1, image2))
    
    Use on RGB image converted from uint8 without normalization:
       msssim_loss = MSSSIM_2D_Objective(luminance=255)
       print('loss')
       print(msssim_loss(image1, image2))
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=(11, 11), 
                 sigmas=(0.5, 1., 2., 4., 8.), luminance=1.0):
        """
        Multi-Scale Structural Similarity (loss function) similar to DSSIM.
        Note : You should add a regularization term like a l1 loss in addition
               to this one.
        
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            sigma: Tuple of SDs of gaussian kernel (default (.5, 1., 2., 4., 8.))
            kernel_size: Size of the sliding window (odd) (default 3)
            luminance: Max image intensity (default 1.0)
        """
        #TODO: Allow passing of custom kernels
        self.__name__ = 'MSSSIM_2D_Objective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.luminance = luminance
        self.c1 = (self.k1 * self.luminance) ** 2
        self.c2 = (self.k2 * self.luminance) ** 2
        self.dim_ordering = K.image_data_format()
        self.channel_dim = -1 if self.dim_ordering == 'channels_last' else 1
        
        # initialize gaussian kernels
        # filter kernel ordering is always
        # (width, height, in_channels, ch_multiplier)
        self.kernels = [K.variable(self.gauss_2d(kernel_size, s)[:,:,None,None]) 
                        for s in sigmas]
        self.kernels = K.concatenate(self.kernels, axis=-1) # stack in ch_multi-
                                                            # plier axis
        self.num = len(sigmas) # number of scales to calculate
    
    def __call__(self, y_true, y_pred):
        '''
        Compute multiscale ssim loss according to Zhao 2016.
        
        It is defined as 1 - msssim().
        Check there for more documentation.
                
        # Arguments
            y_true: Keras Tensor with Rank 4: Image to compare to
            y_pred: Keras Tensor with Rank 4: Image to compare
        '''
        return self.msssim_loss(y_true, y_pred)
    
    def msssim_loss(self, y_true, y_pred):
        '''
        Compute multiscale ssim loss according to Zhao 2016.
        
        It is defined as 1 - msssim().
        Check there for more documentation.
                
        # Arguments
            y_true: Keras Tensor with Rank 4: Image to compare to
            y_pred: Keras Tensor with Rank 4: Image to compare
        '''
        return 1. - self.msssim(y_true, y_pred)
    
    def __int_shape(self, x):
        '''
        Retrun tensor shape as list.
        Selects correct Backend function depending on active Keras backend.
        '''
        return K.int_shape(x) if K.backend() == 'tensorflow' else K.shape(x)
    
    def msssim(self, y_true, y_pred):
        '''
        Compute multiscale ssim according to Zhao 2016.
        
        Has only been tested with tensorflow backend (channels last) so far!
        
        Uses convolutions to do the calculations in one go.
        
        This function takes proper 2D Keras Tensors (NWHC or NCWH)
        
        # Arguments
            y_true: Keras Tensor with Rank 4: Image to compare to
            y_pred: Keras Tensor with Rank 4: Image to compare
        '''
        # some useful inits
        channels = self.__int_shape(y_pred)[self.channel_dim]
                
        # repeat kernel for each channel
        kernel = K.tile(self.kernels, [1, 1, channels, 1])
        
        # compute means
        mu_true = K.depthwise_conv2d(y_true, kernel, padding='same')
        mu_pred = K.depthwise_conv2d(y_pred, kernel, padding='same')
        
        # compute mean squares
        mu_true_sq = K.square(mu_true)
        mu_pred_sq = K.square(mu_pred)
        mu_true_pred = mu_true * mu_pred
        
        # compute input square
        y_true_sq = K.square(y_true)
        y_pred_sq = K.square(y_pred)
        y_true_pred = y_true * y_pred
        
        # compute variances/covariance
        sigma_true_sq = K.depthwise_conv2d(y_true_sq, kernel, padding='same')
        sigma_pred_sq = K.depthwise_conv2d(y_pred_sq, kernel, padding='same')
        sigma_true_pred = K.depthwise_conv2d(y_true_pred, kernel, padding='same')
        
        # centered squares of variances
        sigma_true_sq -= mu_true_sq
        sigma_pred_sq -= mu_pred_sq
        sigma_true_pred -= mu_true_pred
        
        # compute luminance term (l), select only maximum kernel for each channel
        l = (2 * mu_true_pred + self.c1) / (mu_true_sq + mu_pred_sq + self.c1)
        if self.dim_ordering == 'channels_last':
            l_max = l[:,:,:,(self.num - 1)::self.num]
        else:
            l_max = l[:,(self.num - 1)::self.num,:,:]
                
        # compute contrast-structure term (cs)
        cs = (2 * sigma_true_pred + self.c2) / (sigma_true_sq + sigma_pred_sq +
                                                self.c2)
        
        # compute product of different scale cs
        if self.dim_ordering == 'channels_last':
            pcs = [K.prod(cs[:,:,:,i*self.num:(i+1)*self.num], axis=-1, 
                          keepdims=True) for i in range(channels)]
        else:
            pcs = [K.prod(cs[:,i*self.num:(i+1)*self.num,:,:], axis=1,
                          keepdims=True) for i in range(channels)]
        pcs = K.concatenate(pcs, axis=self.channel_dim)
        
        # compute msssim map
        msssim = l_max * pcs # do normalization?
        return msssim
    
    @staticmethod
    def gauss_2d(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask comparable to matlabs fspecial('gaussian', ...)

        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])

        Code from Stack Overflow's thread
        https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-
        filter-in-python

        # Parameters
            shape: size of the returned matrix (tuple of rank 2)
            sigma: the spread of the gauss function
        """
        m, n = [(ss-1.)/2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

