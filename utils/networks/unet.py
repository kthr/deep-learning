from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import BatchNormalization, Concatenate, Conv2D, Conv3D, Dropout, Input, UpSampling2D, UpSampling3D

def UNet_3D(input_shape=(None,None,None,1), depth=5, max_filters=256, kernel_size=(3,3,3), n_output_channels=1, activation='sigmoid', drop_out=0, dropout_at_prediction=False, batch_norm=False):
    '''U net architecture (down/up sampling with skip architecture)
    See: http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Keras implementatio:; https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    def Conv3DReluBatchNorm(n_filters, kernel_size, strides, inputs, drop_out, batch_norm, name=None):
        c = Conv3D(n_filters, kernel_size, strides=strides, padding='same', kernel_initializer='glorot_normal', activation='relu')(inputs)
        if batch_norm:
            c = BatchNormalization(scale=False)(c)
        if drop_out:
            if dropout_at_prediction:
                c = Dropout(rate=drop_out)(c, training=True)
            else:
                c = Dropout(rate=drop_out)(c)
        return c
    
    if max_filters%2!=0:
        raise Exception('max_filters must be divisible by 2!')

    layer_stack=[]
    inputs = Input(input_shape)
    layer = Conv3DReluBatchNorm(int(max_filters/2**depth), kernel_size, (1,1,1), inputs, drop_out, batch_norm, 'conv3D_0')
    layer_stack.append(layer)
    
    for i in range(1,depth+1):
        layer = Conv3DReluBatchNorm(int(max_filters/2**(depth-i)), kernel_size, (2,2,2), layer, drop_out, batch_norm, 'conv3D_'+str(i))
        layer_stack.append(layer)
        
    layer_stack.pop()
    
    for i in range(depth-1,-1,-1):
        layer = Concatenate(axis=-1)([UpSampling3D(size=(2,2,2))(layer), layer_stack.pop()])
        layer = Conv3DReluBatchNorm(int(max_filters/2**(depth-i)), kernel_size, (1,1,1), layer, drop_out, batch_norm, 'conv3D_' + str(depth+(depth-i+1)))
    
    outputs=Conv3D(n_output_channels, (1,1,1), strides=(1,1,1), activation=activation)(layer)
    model = Model(inputs=inputs, outputs=outputs)
                                    
    return model

def UNet_2D(input_shape=(None,None,1), depth=5, max_filters=256, kernel_size=(3,3), n_output_channels=1, activation='sigmoid', drop_out=0, dropout_at_prediction=False, batch_norm=False):
    '''U net architecture (down/up sampling with skip architecture)
    See: http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    Keras implementatio:; https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    def Conv2DReluBatchNorm(n_filters, kernel_size, strides, inputs, drop_out, batch_norm, name=None):
        c = Conv2D(n_filters, kernel_size, strides=strides, padding='same', kernel_initializer='glorot_normal', activation='relu')(inputs)
        if batch_norm:
            c = BatchNormalization(scale=False)(c)
        if drop_out:
            if dropout_at_prediction:
                c = Dropout(rate=drop_out)(c, training=True)
            else:
                c = Dropout(rate=drop_out)(c)
        return c
    
    if max_filters%2!=0:
        raise Exception('max_filters must be divisible by 2!')

    layer_stack=[]
    inputs = Input(input_shape)
    layer = Conv2DReluBatchNorm(int(max_filters/2**depth), kernel_size, (1,1), inputs, drop_out, batch_norm, 'conv2D_0')
    layer_stack.append(layer)
    
    for i in range(1,depth+1):
        layer = Conv2DReluBatchNorm(int(max_filters/2**(depth-i)), kernel_size, (2,2), layer, drop_out, batch_norm, 'conv2D_'+str(i))
        layer_stack.append(layer)
        
    layer_stack.pop()
    
    for i in range(depth-1,-1,-1):
        layer = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer), layer_stack.pop()])
        layer = Conv2DReluBatchNorm(int(max_filters/2**(depth-i)), kernel_size, (1,1), layer, drop_out, batch_norm, 'conv2D_' + str(depth+(depth-i+1)))
    
    outputs=Conv2D(n_output_channels, (1,1), strides=(1,1), activation=activation)(layer)
    model = Model(inputs=inputs, outputs=outputs)
                                    
    return model